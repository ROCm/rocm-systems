// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for the simulator-agnostic hazard_core helpers.

#include <gtest/gtest.h>

#include "detail/access_handling.h"
#include "detail/fifo_ops.h"
#include "detail/wait_handling.h"
#include "detail/waitcnt_decode.h"

#include <deque>
#include <set>
#include <stdexcept>
#include <unordered_map>

using namespace hazard_core;

namespace {

using hazard_core::EntityId;
using hazard_core::PendingAsyncOp;
using hazard_core::PendingRegisterSet;
using hazard_core::WaitCntType;

constexpr hazard_core::VectorRegisterFile kVgpr = hazard_core::VectorRegisterFile::Vector;

const PendingAsyncOp *vgpr_pending(const hazard_core::WaveState &wave, PendingRegisterSet set,
                                   uint32_t reg, EntityId current_instruction_id) {
  return hazard_core::check_vector_hazard_info(&wave, kVgpr, set, reg, 1, current_instruction_id)
      .pending;
}

} // namespace

TEST(HazardCoreAccessHandlingTest, TracksAndChecksVgprWriteReadAndWait) {
  hazard_core::WaveState wave;

  hazard_core::track_vector_write(&wave, kVgpr, 10, 0x100, 4, 2, WaitCntType::VMEM);

  const auto *raw = vgpr_pending(wave, PendingRegisterSet::Writes, 4, 20);
  ASSERT_NE(raw, nullptr);
  EXPECT_EQ(raw->instruction_id, 10u);
  EXPECT_EQ(raw->pc, 0x100u);
  EXPECT_EQ(raw->wait_type, WaitCntType::VMEM);
  EXPECT_NE(hazard_core::check_vector_waw_hazard_info(&wave, kVgpr, 5, 1, 20).pending, nullptr);
  EXPECT_EQ(vgpr_pending(wave, PendingRegisterSet::Writes, 6, 20), nullptr);
  EXPECT_EQ(vgpr_pending(wave, PendingRegisterSet::Writes, 4, 10), nullptr)
      << "same instruction should not self-hazard";

  const auto raw_info =
      hazard_core::check_vector_hazard_info(&wave, kVgpr, PendingRegisterSet::Writes, 3, 4, 20);
  ASSERT_NE(raw_info.pending, nullptr);
  EXPECT_EQ(raw_info.register_index, 4u);
  EXPECT_EQ(raw_info.pending->instruction_id, 10u);

  hazard_core::track_vector_read(&wave, kVgpr, 11, 0x120, 8, 2);
  const auto *war = vgpr_pending(wave, PendingRegisterSet::Reads, 9, 20);
  ASSERT_NE(war, nullptr);
  EXPECT_EQ(war->instruction_id, 11u);
  EXPECT_EQ(war->wait_type, WaitCntType::STORE);

  hazard_core::process_wait_instruction(&wave, WaitCntType::VMEM, 0);
  EXPECT_EQ(vgpr_pending(wave, PendingRegisterSet::Writes, 4, 20), nullptr);
  EXPECT_NE(vgpr_pending(wave, PendingRegisterSet::Reads, 8, 20), nullptr);

  hazard_core::process_wait_instruction(&wave, WaitCntType::STORE, 0);
  EXPECT_EQ(vgpr_pending(wave, PendingRegisterSet::Reads, 8, 20), nullptr);
}

TEST(WaitcntDecodeTest, DecodesLegacyWaitcntSplitVmcntBits) {
  constexpr uint32_t simm16 = 0x4000u | (5u << 8) | 7u;

  const auto fields = data_hazard_waitcnt::decode_legacy_waitcnt(
      simm16, data_hazard_waitcnt::LEGACY_LGKMCNT_5BIT_MASK);

  EXPECT_EQ(fields.vmcnt, 0x17u);
  EXPECT_EQ(fields.lgkmcnt, 5u);
}

TEST(WaitcntDecodeTest, DecodesLegacyWaitcntWithFourBitLgkmcntMask) {
  constexpr uint32_t simm16 = (0x1Fu << 8) | 3u;

  const auto fields = data_hazard_waitcnt::decode_legacy_waitcnt(
      simm16, data_hazard_waitcnt::LEGACY_LGKMCNT_4BIT_MASK);

  EXPECT_EQ(fields.vmcnt, 3u);
  EXPECT_EQ(fields.lgkmcnt, 0xFu);
}

TEST(WaitcntDecodeTest, DecodesSplitWaitcntPrimaryAndDscntFields) {
  constexpr uint32_t simm16 = (9u << 8) | 4u;

  const auto fields = data_hazard_waitcnt::decode_split_waitcnt(simm16);

  EXPECT_EQ(fields.primary, 9u);
  EXPECT_EQ(fields.dscnt, 4u);
}

TEST(WaitcntDecodeTest, SplitWaitcntEncodeRoundTrips) {
  const uint32_t simm16 = data_hazard_waitcnt::encode_split_waitcnt(9u, 4u);

  const auto fields = data_hazard_waitcnt::decode_split_waitcnt(simm16);

  EXPECT_EQ(simm16, (9u << 8) | 4u);
  EXPECT_EQ(fields.primary, 9u);
  EXPECT_EQ(fields.dscnt, 4u);
}

TEST(WaitcntDecodeTest, LegacyWaitcntEncodeRoundTripsSplitVmcntBits) {
  const uint32_t simm16 = data_hazard_waitcnt::encode_legacy_waitcnt(
      0x17u, 5u, data_hazard_waitcnt::LEGACY_LGKMCNT_4BIT_MASK);

  const auto fields = data_hazard_waitcnt::decode_legacy_waitcnt(
      simm16, data_hazard_waitcnt::LEGACY_LGKMCNT_4BIT_MASK);

  EXPECT_EQ(fields.vmcnt, 0x17u);
  EXPECT_EQ(fields.lgkmcnt, 5u);
}

TEST(HazardCoreAccessHandlingTest, TracksSgprWritesAndWait) {
  hazard_core::WaveState wave;

  hazard_core::track_sgpr_write(&wave, 12, 0x200, 20, WaitCntType::SMEM, 2);

  const auto *raw = hazard_core::check_sgpr_read_hazard(&wave, 21, 1, 30);
  ASSERT_NE(raw, nullptr);
  EXPECT_EQ(raw->instruction_id, 12u);
  EXPECT_EQ(raw->wait_type, WaitCntType::SMEM);
  EXPECT_EQ(hazard_core::check_sgpr_read_hazard(&wave, 22, 1, 30), nullptr);

  hazard_core::process_wait_instruction(&wave, WaitCntType::SMEM, 0);
  EXPECT_EQ(hazard_core::check_sgpr_read_hazard(&wave, 20, 1, 30), nullptr);
}

TEST(HazardCoreAccessHandlingTest, TracksLdsReadWriteAndTensorRanges) {
  hazard_core::WaveState wave;

  hazard_core::track_lds_write(&wave, 13, 0x300, 0x80, 8);
  const auto *raw = hazard_core::check_lds_read_hazard(&wave, 0x84, 4);
  ASSERT_NE(raw, nullptr);
  EXPECT_EQ(raw->instruction_id, 13u);

  hazard_core::track_lds_read(&wave, 14, 0x304, 0x100, 16);
  const auto *war = hazard_core::check_lds_write_hazard(&wave, 0x108, 4);
  ASSERT_NE(war, nullptr);
  EXPECT_EQ(war->instruction_id, 14u);

  hazard_core::track_tensor_lds(&wave, 15, 0x308, 0x180, 32);
  const auto *tensor = hazard_core::check_tensor_lds_hazard(&wave, 0x190, 4);
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->instruction_id, 15u);
  EXPECT_EQ(tensor->wait_type, WaitCntType::TENSOR);

  hazard_core::process_wait_instruction(&wave, WaitCntType::LDS, 0);
  EXPECT_EQ(hazard_core::check_lds_read_hazard(&wave, 0x80, 4), nullptr);
  EXPECT_EQ(hazard_core::check_lds_write_hazard(&wave, 0x100, 4), nullptr);
  EXPECT_NE(hazard_core::check_tensor_lds_hazard(&wave, 0x180, 4), nullptr);

  hazard_core::process_wait_instruction(&wave, WaitCntType::TENSOR, 0);
  EXPECT_EQ(hazard_core::check_tensor_lds_hazard(&wave, 0x180, 4), nullptr);
}

TEST(HazardCoreAccessHandlingTest, LdsWaitKeepsAllEntriesWhenKeepCountExceedsPendingWrites) {
  hazard_core::WaveState wave;

  hazard_core::track_lds_write(&wave, 13, 0x300, 0x80, 8);
  hazard_core::track_lds_write(&wave, 14, 0x304, 0x100, 16);

  hazard_core::process_wait_instruction(&wave, WaitCntType::LDS, 4);

  EXPECT_EQ(wave.lds_fifo.size(), 2u);
  EXPECT_NE(hazard_core::check_lds_read_hazard(&wave, 0x84, 4), nullptr);
  EXPECT_NE(hazard_core::check_lds_read_hazard(&wave, 0x108, 4), nullptr);
}

TEST(HazardCoreAccessHandlingTest, TracksFlatDsSideSeparately) {
  hazard_core::WaveState wave;

  hazard_core::track_vector_flat_ds(&wave, kVgpr, 16, 0x400, 30, 2, true);

  const auto *pending = vgpr_pending(wave, PendingRegisterSet::DsWrites, 31, 20);
  ASSERT_NE(pending, nullptr);
  EXPECT_EQ(pending->instruction_id, 16u);
  EXPECT_TRUE(pending->is_flat_ds);
  EXPECT_EQ(vgpr_pending(wave, PendingRegisterSet::Writes, 31, 20), nullptr);

  hazard_core::process_wait_instruction(&wave, WaitCntType::LDS, 0);
  EXPECT_EQ(vgpr_pending(wave, PendingRegisterSet::DsWrites, 31, 20), nullptr);
}

TEST(HazardCoreAccessHandlingTest, IteratesPendingInstructionIdsAcrossCoreState) {
  hazard_core::WaveState wave;

  hazard_core::track_vector_write(&wave, kVgpr, 10, 0x100, 4, 1, WaitCntType::VMEM);
  hazard_core::track_sgpr_write(&wave, 11, 0x104, 8, WaitCntType::SMEM);
  hazard_core::track_vector_flat_ds(&wave, kVgpr, 12, 0x108, 16, 1, true);
  hazard_core::track_vector_read(&wave, kVgpr, 13, 0x10c, 20, 1);
  hazard_core::track_lds_write(&wave, 14, 0x110, 0x80, 4);
  hazard_core::track_lds_read(&wave, 15, 0x114, 0x90, 4);
  hazard_core::track_tensor_lds(&wave, 16, 0x118, 0xa0, 4);

  std::set<EntityId> ids;
  hazard_core::for_each_pending_instruction_id(&wave, [&](EntityId id) { ids.insert(id); });

  EXPECT_EQ(ids, (std::set<EntityId>{10, 11, 12, 13, 14, 15, 16}));

  uint32_t null_visits = 0;
  hazard_core::for_each_pending_instruction_id(nullptr, [&](EntityId) { ++null_visits; });
  EXPECT_EQ(null_visits, 0u);
}

class ClearRegisterFifoTest : public ::testing::Test {
protected:
  using Fifo = std::deque<std::pair<uint32_t, PendingAsyncOp>>;
  using Map = std::unordered_map<uint32_t, PendingAsyncOp>;

  static PendingAsyncOp make_op(EntityId id) { return {id, 0, WaitCntType::VMEM, 4}; }

  void push(Fifo &fifo, Map &map, uint32_t reg, EntityId id) {
    auto op = make_op(id);
    map[reg] = op;
    fifo.push_back({reg, op});
  }
};

TEST_F(ClearRegisterFifoTest, KeepZeroClearsEverything) {
  Fifo fifo;
  Map map;
  push(fifo, map, 0, 1);
  push(fifo, map, 1, 2);
  push(fifo, map, 2, 3);

  clear_register_fifo(fifo, map, 0);

  EXPECT_TRUE(fifo.empty());
  EXPECT_TRUE(map.empty());
}

TEST_F(ClearRegisterFifoTest, KeepNRetainsNewestEntries) {
  Fifo fifo;
  Map map;
  push(fifo, map, 10, 1);
  push(fifo, map, 11, 2);
  push(fifo, map, 12, 3);

  clear_register_fifo(fifo, map, 2);

  EXPECT_EQ(fifo.size(), 2u);
  EXPECT_EQ(fifo.front().first, 11u);
  EXPECT_EQ(fifo.back().first, 12u);
  EXPECT_EQ(map.count(10), 0u);
  EXPECT_EQ(map.count(11), 1u);
  EXPECT_EQ(map.count(12), 1u);
}

TEST_F(ClearRegisterFifoTest, RepeatedRegisterKeepsMapEntryIfNewestEntryRemains) {
  Fifo fifo;
  Map map;
  push(fifo, map, 7, 1);
  push(fifo, map, 8, 2);
  push(fifo, map, 7, 3);

  clear_register_fifo(fifo, map, 2);

  EXPECT_EQ(fifo.size(), 2u);
  EXPECT_EQ(map.count(7), 1u);
  EXPECT_EQ(map[7].instruction_id, 3u);
  EXPECT_EQ(map.count(8), 1u);
}

TEST(HazardCoreWaitHandlingTest, FindsLdsWriteOverlapAdjacentAndDisjointRanges) {
  hazard_core::WaveState wave;
  wave.lds_fifo.push_back({0x40, PendingAsyncOp{1, 0x100, WaitCntType::LDS, 8}});

  EXPECT_EQ(hazard_core::find_pending_lds_write(
                static_cast<const hazard_core::WaveState *>(nullptr), 0x40, 4),
            nullptr);
  EXPECT_NE(hazard_core::find_pending_lds_write(&wave, 0x40, 4), nullptr);
  EXPECT_NE(hazard_core::find_pending_lds_write(&wave, 0x44, 4), nullptr);
  EXPECT_EQ(hazard_core::find_pending_lds_write(&wave, 0x48, 4), nullptr);
  EXPECT_EQ(hazard_core::find_pending_lds_write(&wave, 0x100, 4), nullptr);
}

TEST(HazardCoreWaitHandlingTest, FindsTensorLdsWriteOverlapAdjacentAndDisjointRanges) {
  hazard_core::WaveState wave;
  wave.tensor_lds_fifo.push_back({0x200, PendingAsyncOp{10, 0x100, WaitCntType::TENSOR, 16}});

  EXPECT_EQ(hazard_core::find_pending_tensor_lds_write(
                static_cast<const hazard_core::WaveState *>(nullptr), 0x200, 4),
            nullptr);
  EXPECT_EQ(hazard_core::find_pending_tensor_lds_write(&wave, 0x200, 16),
            &wave.tensor_lds_fifo.front().second);
  EXPECT_EQ(hazard_core::find_pending_tensor_lds_write(&wave, 0x208, 8),
            &wave.tensor_lds_fifo.front().second);
  EXPECT_EQ(hazard_core::find_pending_tensor_lds_write(&wave, 0x210, 4), nullptr);
  EXPECT_EQ(hazard_core::find_pending_tensor_lds_write(&wave, 0x180, 4), nullptr);
}

TEST(HazardCoreWaitHandlingTest, FindsLdsReadOverlapAdjacentAndDisjointRanges) {
  hazard_core::WaveState wave;
  wave.lds_read_fifo.push_back({0x300, PendingAsyncOp{20, 0x150, WaitCntType::LDS, 8}});

  EXPECT_EQ(hazard_core::find_pending_lds_read(static_cast<const hazard_core::WaveState *>(nullptr),
                                               0x300, 4),
            nullptr);
  EXPECT_EQ(hazard_core::find_pending_lds_read(&wave, 0x300, 8),
            &wave.lds_read_fifo.front().second);
  EXPECT_EQ(hazard_core::find_pending_lds_read(&wave, 0x304, 8),
            &wave.lds_read_fifo.front().second);
  EXPECT_EQ(hazard_core::find_pending_lds_read(&wave, 0x308, 4), nullptr);
  EXPECT_EQ(hazard_core::find_pending_lds_read(&wave, 0x200, 4), nullptr);
}

TEST(HazardCoreWaitHandlingTest, LdsRangeChecksUseWideMathNearUint32Max) {
  hazard_core::WaveState wave;
  wave.lds_fifo.push_back({0xfffffff0u, PendingAsyncOp{1, 0x100, WaitCntType::LDS, 32}});
  wave.tensor_lds_fifo.push_back({0xfffffff0u, PendingAsyncOp{2, 0x104, WaitCntType::TENSOR, 32}});
  wave.lds_read_fifo.push_back({0xfffffff0u, PendingAsyncOp{3, 0x108, WaitCntType::LDS, 32}});

  EXPECT_NE(hazard_core::find_pending_lds_write(&wave, 0xfffffff8u, 16), nullptr);
  EXPECT_NE(hazard_core::find_pending_tensor_lds_write(&wave, 0xfffffff8u, 16), nullptr);
  EXPECT_NE(hazard_core::find_pending_lds_read(&wave, 0xfffffff8u, 16), nullptr);
}

class ClearPendingOpsTest : public ::testing::Test {
protected:
  hazard_core::WaveState wave;

  static PendingAsyncOp make_op(EntityId id, WaitCntType type) { return {id, 0, type, 4}; }

  void push_vmem_load(uint32_t reg, EntityId id) {
    auto op = make_op(id, WaitCntType::VMEM);
    wave.pending_vgpr_writes[reg] = op;
    wave.vmem_load_fifo.push_back({reg, op});
  }

  void push_smem_load(uint32_t reg, EntityId id) {
    auto op = make_op(id, WaitCntType::SMEM);
    wave.pending_sgpr_writes[reg] = op;
    wave.smem_load_fifo.push_back({reg, op});
  }

  void push_store(uint32_t reg, EntityId id) {
    auto op = make_op(id, WaitCntType::STORE);
    wave.pending_vgpr_reads[reg] = op;
    wave.vmem_store_fifo.push_back({reg, op});
  }

  void push_lds(uint32_t addr, EntityId id, WaitCntType type = WaitCntType::LDS) {
    auto op = make_op(id, type);
    wave.lds_fifo.push_back({addr, op});
  }

  /// A store as the frontends report it: a slot of the counter it is
  /// outstanding on and nothing else, its data sources having been captured
  /// when it issued. VMEM is the gfx9 and CDNA counter, STORE the gfx10-and-
  /// later one.
  void push_store_op(EntityId id, WaitCntType counter) {
    wave.vmem_store_ops.push_back(make_op(id, counter));
  }
};

TEST_F(ClearPendingOpsTest, NullWaveIsNoop) {
  EXPECT_NO_THROW(hazard_core::clear_pending_ops(nullptr, WaitCntType::VMEM, 0));
}

TEST_F(ClearPendingOpsTest, ClearsRegisterFifosByWaitType) {
  push_vmem_load(0, 1);
  push_smem_load(1, 2);
  push_store(2, 3);

  hazard_core::clear_pending_ops(&wave, WaitCntType::VMEM, 0);
  EXPECT_TRUE(wave.vmem_load_fifo.empty());
  EXPECT_TRUE(wave.pending_vgpr_writes.empty());
  EXPECT_EQ(wave.smem_load_fifo.size(), 1u);
  EXPECT_EQ(wave.vmem_store_fifo.size(), 1u);

  hazard_core::clear_pending_ops(&wave, WaitCntType::SMEM, 0);
  EXPECT_TRUE(wave.smem_load_fifo.empty());
  EXPECT_TRUE(wave.pending_sgpr_writes.empty());

  hazard_core::clear_pending_ops(&wave, WaitCntType::STORE, 0);
  EXPECT_TRUE(wave.vmem_store_fifo.empty());
  EXPECT_TRUE(wave.pending_vgpr_reads.empty());
}

TEST_F(ClearPendingOpsTest, NonzeroSmemWaitDoesNotClearOutOfOrderScalarLoads) {
  push_smem_load(1, 1);
  push_smem_load(2, 2);

  hazard_core::clear_pending_ops(&wave, WaitCntType::SMEM, 1);

  EXPECT_EQ(wave.smem_load_fifo.size(), 2u);
  EXPECT_EQ(wave.pending_sgpr_writes.size(), 2u);

  hazard_core::clear_pending_ops(&wave, WaitCntType::SMEM, 0);

  EXPECT_TRUE(wave.smem_load_fifo.empty());
  EXPECT_TRUE(wave.pending_sgpr_writes.empty());
}

TEST_F(ClearPendingOpsTest, KeepsNewestEntriesForWaitN) {
  push_vmem_load(0, 1);
  push_vmem_load(1, 2);
  push_vmem_load(2, 3);

  hazard_core::clear_pending_ops(&wave, WaitCntType::VMEM, 1);

  EXPECT_EQ(wave.vmem_load_fifo.size(), 1u);
  EXPECT_EQ(wave.vmem_load_fifo.front().first, 2u);
  EXPECT_EQ(wave.pending_vgpr_writes.count(2), 1u);
}

TEST_F(ClearPendingOpsTest, WaitNCountsInstructionsNotDestinationRegisters) {
  // global_load_dwordx4 v[0:3] leaves four FIFO entries but occupies a single
  // LOADcnt slot, so s_wait_loadcnt 1 leaves the whole load outstanding.
  for (uint32_t reg = 0; reg < 4; ++reg)
    push_vmem_load(reg, 1);

  hazard_core::clear_pending_ops(&wave, WaitCntType::VMEM, 1);

  EXPECT_EQ(wave.vmem_load_fifo.size(), 4u);
  EXPECT_EQ(wave.pending_vgpr_writes.size(), 4u);
}

TEST_F(ClearPendingOpsTest, WaitNRetiresEveryRegisterOfTheOperationsItDrains) {
  for (uint32_t reg = 0; reg < 4; ++reg)
    push_vmem_load(reg, 1);
  for (uint32_t reg = 4; reg < 8; ++reg)
    push_vmem_load(reg, 2);

  hazard_core::clear_pending_ops(&wave, WaitCntType::VMEM, 1);

  // The older load retires whole; the newer one stays whole.
  EXPECT_EQ(wave.vmem_load_fifo.size(), 4u);
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(wave.pending_vgpr_writes.count(reg), 0u) << "register " << reg;
  for (uint32_t reg = 4; reg < 8; ++reg)
    EXPECT_EQ(wave.pending_vgpr_writes.count(reg), 1u) << "register " << reg;
}

TEST_F(ClearPendingOpsTest, SmemWaitNCountsInstructionsNotDestinationRegisters) {
  // s_load_dwordx8 s[0:7]: one KMcnt slot, eight FIFO entries.
  for (uint32_t reg = 0; reg < 8; ++reg)
    push_smem_load(reg, 1);
  for (uint32_t reg = 8; reg < 16; ++reg)
    push_smem_load(reg, 2);

  hazard_core::clear_pending_ops(&wave, WaitCntType::SMEM, 1);

  // Scalar loads complete out of order, so a partial wait drains nothing at all.
  EXPECT_EQ(wave.smem_load_fifo.size(), 16u);
  EXPECT_EQ(wave.pending_sgpr_writes.size(), 16u);
}

TEST_F(ClearPendingOpsTest, StoreWaitNCountsInstructionsNotSourceRegisters) {
  for (uint32_t reg = 0; reg < 4; ++reg)
    push_store(reg, 1);
  for (uint32_t reg = 4; reg < 8; ++reg)
    push_store(reg, 2);

  hazard_core::clear_pending_ops(&wave, WaitCntType::STORE, 1);

  EXPECT_EQ(wave.vmem_store_fifo.size(), 4u);
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(wave.pending_vgpr_reads.count(reg), 0u) << "register " << reg;
  for (uint32_t reg = 4; reg < 8; ++reg)
    EXPECT_EQ(wave.pending_vgpr_reads.count(reg), 1u) << "register " << reg;
}

TEST_F(ClearPendingOpsTest, StoreWaitDrainsCounterSlotsHoldingNoRegisterState) {
  push_store_op(1, WaitCntType::STORE);
  push_store_op(2, WaitCntType::STORE);
  push_store_op(3, WaitCntType::STORE);

  hazard_core::clear_pending_ops(&wave, WaitCntType::STORE, 1);

  ASSERT_EQ(wave.vmem_store_ops.size(), 1u);
  EXPECT_EQ(wave.vmem_store_ops.front().instruction_id, 3u);
}

TEST_F(ClearPendingOpsTest, VmemWaitCountsStoresOfItsOwnCounterAgainstTheLoadsItKeeps) {
  // The gfx9 sequence a scratch spill produces: one global load, then two
  // scratch stores, then s_waitcnt vmcnt(2). One counter holds all three in
  // issue order, so keeping two retires the load and the read of its
  // destination that follows is safe.
  push_vmem_load(0, 1);
  push_store_op(2, WaitCntType::VMEM);
  push_store_op(3, WaitCntType::VMEM);

  hazard_core::clear_pending_ops(&wave, WaitCntType::VMEM, 2);

  EXPECT_TRUE(wave.vmem_load_fifo.empty());
  EXPECT_TRUE(wave.pending_vgpr_writes.empty());
  EXPECT_EQ(wave.vmem_store_ops.size(), 2u);
}

TEST_F(ClearPendingOpsTest, VmemWaitLeavesLoadPendingWhenTooFewOperationsPrecedeIt) {
  // The same wait with one store fewer keeps two operations outstanding, so
  // the load has not landed and reading its destination is still a hazard.
  push_vmem_load(0, 1);
  push_store_op(2, WaitCntType::VMEM);

  hazard_core::clear_pending_ops(&wave, WaitCntType::VMEM, 2);

  EXPECT_EQ(wave.vmem_load_fifo.size(), 1u);
  EXPECT_EQ(wave.pending_vgpr_writes.count(0), 1u);
  EXPECT_EQ(wave.vmem_store_ops.size(), 1u);
}

TEST_F(ClearPendingOpsTest, VmemWaitIgnoresStoresCountedOnTheirOwnCounter) {
  // gfx10 and gfx11 count stores on vscnt, so the same three operations leave
  // vmcnt(2) with one load outstanding and nothing to retire. The stores wait
  // for s_waitcnt_vscnt instead.
  push_vmem_load(0, 1);
  push_store_op(2, WaitCntType::STORE);
  push_store_op(3, WaitCntType::STORE);

  hazard_core::clear_pending_ops(&wave, WaitCntType::VMEM, 2);

  EXPECT_EQ(wave.vmem_load_fifo.size(), 1u);
  EXPECT_EQ(wave.pending_vgpr_writes.count(0), 1u);
  EXPECT_EQ(wave.vmem_store_ops.size(), 2u);

  hazard_core::clear_pending_ops(&wave, WaitCntType::STORE, 0);
  EXPECT_TRUE(wave.vmem_store_ops.empty());
  EXPECT_EQ(wave.vmem_load_fifo.size(), 1u);
}

TEST_F(ClearPendingOpsTest, StoreWaitLeavesStoresOfTheLoadCounterOutstanding) {
  // An explicit store wait on a target whose stores are counted with its loads
  // has nothing of its own to drain.
  push_store_op(1, WaitCntType::VMEM);

  hazard_core::clear_pending_ops(&wave, WaitCntType::STORE, 0);

  EXPECT_EQ(wave.vmem_store_ops.size(), 1u);
}

TEST_F(ClearPendingOpsTest, VmemWaitNCombinesRegisterAndLdsOperations) {
  push_vmem_load(0, 1);
  push_lds(0x10, 2, WaitCntType::VMEM);
  push_vmem_load(1, 3);
  push_lds(0x20, 4, WaitCntType::VMEM);

  hazard_core::clear_pending_ops(&wave, WaitCntType::VMEM, 1);

  EXPECT_TRUE(wave.vmem_load_fifo.empty());
  EXPECT_TRUE(wave.pending_vgpr_writes.empty());
  ASSERT_EQ(wave.lds_fifo.size(), 1u);
  EXPECT_EQ(wave.lds_fifo.front().first, 0x20u);
  EXPECT_EQ(wave.lds_fifo.front().second.instruction_id, 4u);
}

TEST_F(ClearPendingOpsTest, ClearsLdsWriteReadAndFlatDsState) {
  push_lds(0x40, 1);
  wave.lds_read_fifo.push_back({0x80, make_op(2, WaitCntType::LDS)});
  hazard_core::track_vector_flat_ds(&wave, kVgpr, 3, 0x100, 4, 1);

  hazard_core::clear_pending_ops(&wave, WaitCntType::LDS, 0);

  EXPECT_TRUE(wave.lds_fifo.empty());
  EXPECT_TRUE(wave.lds_read_fifo.empty());
  EXPECT_TRUE(wave.flat_vgpr_ds_fifo.empty());
  EXPECT_TRUE(wave.pending_vgpr_writes_ds.empty());
}

TEST_F(ClearPendingOpsTest, ClearsTensorStateOnlyForTensorWait) {
  wave.tensor_lds_fifo.push_back({0x100, make_op(4, WaitCntType::TENSOR)});
  push_lds(0x40, 1);

  hazard_core::clear_pending_ops(&wave, WaitCntType::TENSOR, 0);

  EXPECT_TRUE(wave.tensor_lds_fifo.empty());
  EXPECT_EQ(wave.lds_fifo.size(), 1u);
}

TEST_F(ClearPendingOpsTest, ExplicitUnsupportedTypes) {
  // Runs on the simulator callback path, so unsupported counters are reported
  // through the return value instead of thrown.
  EXPECT_TRUE(hazard_core::clear_pending_ops(&wave, WaitCntType::XCNT, 0));
  EXPECT_FALSE(hazard_core::clear_pending_ops(&wave, WaitCntType::NONE, 0));
  EXPECT_FALSE(hazard_core::clear_pending_ops(&wave, WaitCntType::ASYNC, 0));
}

TEST_F(ClearPendingOpsTest, LDSFifoKeepNPreservesNewestMatchingWaitTypeOnly) {
  push_lds(0x10, 1, WaitCntType::VMEM);
  push_lds(0x20, 2, WaitCntType::LDS);
  push_lds(0x30, 3, WaitCntType::VMEM);
  push_lds(0x40, 4, WaitCntType::LDS);
  push_lds(0x50, 5, WaitCntType::VMEM);

  clear_pending_ops(&wave, WaitCntType::VMEM, 1);

  ASSERT_EQ(wave.lds_fifo.size(), 3u);
  EXPECT_EQ(wave.lds_fifo[0].first, 0x20u);
  EXPECT_EQ(wave.lds_fifo[0].second.wait_type, WaitCntType::LDS);
  EXPECT_EQ(wave.lds_fifo[1].first, 0x40u);
  EXPECT_EQ(wave.lds_fifo[1].second.wait_type, WaitCntType::LDS);
  EXPECT_EQ(wave.lds_fifo[2].first, 0x50u);
  EXPECT_EQ(wave.lds_fifo[2].second.wait_type, WaitCntType::VMEM);
}

TEST_F(ClearPendingOpsTest, TypesDoNotCrossContaminate) {
  push_vmem_load(0, 1);
  push_smem_load(0, 2);
  push_store(0, 3);
  push_lds(0x40, 4);

  clear_pending_ops(&wave, WaitCntType::VMEM, 0);

  EXPECT_TRUE(wave.vmem_load_fifo.empty());
  EXPECT_EQ(wave.smem_load_fifo.size(), 1u);
  EXPECT_EQ(wave.vmem_store_fifo.size(), 1u);
  EXPECT_EQ(wave.lds_fifo.size(), 1u);
}

TEST(GetWaitSuggestionTest, ReturnsExpectedSuggestionsAndEmptyForUnsupportedTypes) {
  EXPECT_EQ(hazard_core::get_wait_suggestion(WaitCntType::VMEM),
            "Add s_wait_loadcnt 0 before reading this register");
  EXPECT_EQ(hazard_core::get_wait_suggestion(WaitCntType::SMEM),
            "Add s_wait_kmcnt 0 before reading this register");
  EXPECT_EQ(hazard_core::get_wait_suggestion(WaitCntType::LDS),
            "Add s_wait_dscnt 0 before reading this register");
  EXPECT_EQ(hazard_core::get_wait_suggestion(WaitCntType::STORE),
            "Add s_wait_storecnt 0 before writing this register");
  EXPECT_EQ(hazard_core::get_wait_suggestion(WaitCntType::TENSOR),
            "Add s_wait_tensorcnt 0 before reading this LDS address");
  EXPECT_EQ(hazard_core::get_wait_suggestion(WaitCntType::XCNT), "");
  EXPECT_EQ(hazard_core::get_wait_suggestion(WaitCntType::NONE), "");
  EXPECT_EQ(hazard_core::get_wait_suggestion(WaitCntType::ASYNC), "");
}

TEST(MakeWaitSuggestionTest, NamesTheResourceTheHazardNames) {
  using hazard_core::HazardAccessKind;
  using hazard_core::HazardResourceLabel;

  // A VGPR destination of a DS load is a register hazard even though DScnt
  // resolves it, so the suggestion must not talk about an LDS address.
  EXPECT_EQ(hazard_core::make_wait_suggestion(WaitCntType::LDS, HazardAccessKind::Read,
                                              HazardResourceLabel::Register),
            "Add s_wait_dscnt 0 before reading this register");
  EXPECT_EQ(hazard_core::make_wait_suggestion(WaitCntType::LDS, HazardAccessKind::Write,
                                              HazardResourceLabel::LdsAddress),
            "Add s_wait_dscnt 0 before writing this LDS address");
  EXPECT_EQ(hazard_core::make_wait_suggestion(WaitCntType::VMEM, HazardAccessKind::Write,
                                              HazardResourceLabel::LdsAddress),
            "Add s_wait_loadcnt 0 before writing this LDS address");
  EXPECT_EQ(hazard_core::make_wait_suggestion(WaitCntType::XCNT, HazardAccessKind::Read,
                                              HazardResourceLabel::Register),
            "");
}
