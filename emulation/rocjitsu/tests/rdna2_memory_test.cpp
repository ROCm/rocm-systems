// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "execution_plugin_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

namespace rocjitsu {
namespace {

class Rdna2MemoryTest : public testing::Test {
protected:
  void SetUp() override {
    cache_.set_backing_memory(&memory_);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_RDNA2;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256; // The wave below requests 16 registers from this block.
    config.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu_, "rdna2", config, &memory_, &cache_));
    ASSERT_TRUE(cu_->set_gws_device(gds_));
    decoder_ = Decoder::create(config.arch);
    ASSERT_TRUE(decoder_);
    wave_ = cu_->dispatch_wf(0, 0, 106, 16, 64);
    ASSERT_TRUE(wave_);
    wave_->set_exec(1);
    wave_->set_lds_base(1024);
    wave_->set_lds_size(1024);
  }
  void TearDown() override {
    if (wave_)
      wave_->halt();
  }
  auto execute(const std::array<uint32_t, 2> &words) {
    auto result = decoder_->decode_window(words);
    EXPECT_TRUE(result.succeeded());
    if (result.failed())
      return std::unique_ptr<Instruction>{};
    auto inst = std::move(result).value();
    EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wave_).succeeded());
    return inst;
  }
  void issue(const std::array<uint32_t, 2> &words, amdgpu::MemoryPipeline &pipeline) {
    auto inst = execute(words);
    ASSERT_TRUE(inst && inst->data());
    EXPECT_EQ(pipeline.issue(inst.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
  }
  void write_scalar(unsigned reg, uint32_t value) {
    cu_->write_sgpr(wave_->sgpr_alloc().base + reg, value);
  }
  void write_vector(unsigned reg, uint32_t value, unsigned lane = 0) {
    cu_->write_vgpr(wave_->vgpr_alloc().base + reg, lane, value);
  }
  uint32_t read_vector(unsigned reg, unsigned lane = 0) {
    return cu_->read_vgpr(wave_->vgpr_alloc().base + reg, lane);
  }
  amdgpu::GpuMemory memory_{"test"};
  amdgpu::L2Cache cache_{"test"};
  std::shared_ptr<amdgpu::GwsDevice> gds_ = std::make_shared<amdgpu::GwsDevice>();
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
};

TEST_F(Rdna2MemoryTest, WorkgroupAllocationReservesComplete1024ByteBlocks) {
  EXPECT_EQ(cu_->allocate_lds(1), 0u);
  EXPECT_EQ(cu_->allocate_lds(1025), 1024u);
  EXPECT_EQ(cu_->allocate_lds(1), 3072u);
  EXPECT_EQ(amdgpu::lds_allocation_bytes(cu_->arch(), UINT32_MAX), uint64_t{1} << 32);
}

TEST_F(Rdna2MemoryTest, UnsupportedGdsReadAddtidRejectsAllocatedAndUnallocatedResults) {
  for (uint8_t dst : {8, 16}) {
    SCOPED_TRACE(unsigned(dst));
    auto decoded =
        decoder_->decode_window(rdna2::build_ds(rdna2::kDsReadAddtidB32, {.gds = 1, .vdst = dst}));
    ASSERT_TRUE(decoded.succeeded());
    wave_->clear_instruction_execution_error();
    EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wave_).failed());
    EXPECT_EQ(wave_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(decoded.value()->data(), nullptr);
  }
}

TEST_F(Rdna2MemoryTest, ScalarAddressComponentsHaveDistinctAlignmentRules) {
  write_scalar(0, 0x4002);
  write_scalar(1, 0);
  write_scalar(2, 2);
  auto load = execute(
      rdna2::build_smem(rdna2::kSLoadDword, {.sbase = 0, .sdata = 8, .offset = 2, .soffset = 2}));
  ASSERT_TRUE(load && load->data());
  EXPECT_EQ(load->data_as<amdgpu::ScalarMemState>()->addr, 0x4000u);
  write_scalar(2, 256);
  write_scalar(4, 2);
  load = execute(rdna2::build_smem(rdna2::kSBufferLoadDword,
                                   {.sbase = 0, .sdata = 8, .offset = 0, .soffset = 4}));
  ASSERT_TRUE(load && load->data());
  EXPECT_EQ(load->data_as<amdgpu::ScalarMemState>()->addr, 0x4000u);
}

TEST_F(Rdna2MemoryTest, PermuteIsLocalToEachHalfOfWave64) {
  wave_->set_exec(~uint64_t{0});
  for (unsigned lane = 0; lane < 64; ++lane) {
    write_vector(0, 0, lane);
    write_vector(1, 100 + lane, lane);
  }
  ASSERT_TRUE(execute(rdna2::build_ds(rdna2::kDsBpermuteB32, {.addr = 0, .data0 = 1, .vdst = 2})));
  for (unsigned lane = 0; lane < 64; ++lane)
    EXPECT_EQ(read_vector(2, lane), lane < 32 ? 100u : 132u);
}

TEST_F(Rdna2MemoryTest, DualStoresAndAtomicsRespectWorkgroupBounds) {
  amdgpu::LocalMemPipeline local;
  write_vector(0, 1020);
  write_vector(1, 0x1111);
  write_vector(2, 0x2222);
  wave_->lds().write32(2048, 0xcccc);
  issue(rdna2::build_ds(rdna2::kDsWrite2B32, {.offset1 = 1, .addr = 0, .data0 = 1, .data1 = 2}),
        local);
  EXPECT_EQ(wave_->lds().read32(2044), 0x1111u);
  EXPECT_EQ(wave_->lds().read32(2048), 0xccccu);
  write_vector(0, 1024);
  issue(rdna2::build_ds(rdna2::kDsAddRtnU32, {.addr = 0, .data0 = 1, .vdst = 3}), local);
  EXPECT_EQ(read_vector(3), 0u);
  EXPECT_EQ(wave_->lds().read32(2048), 0xccccu);
  write_vector(0, 0);
  issue(rdna2::build_ds(rdna2::kDsWrite2B32, {.addr = 0, .data0 = 1, .data1 = 2}), local);
  EXPECT_EQ(wave_->lds().read32(1024), 0x1111u);
}

TEST_F(Rdna2MemoryTest, BufferLoadChecksWholeVgprTupleBeforeIssuingMemoryRequests) {
  struct Load {
    bool typed;
    uint16_t opcode;
    bool packed_d16;
    unsigned data_registers;
    uint32_t first_value;
  };
  constexpr Load loads[] = {
      {false, rdna2::kBufferLoadDword, false, 1, 0x76543210},
      {false, rdna2::kBufferLoadDwordx4, false, 4, 0x76543210},
      {false, rdna2::kBufferLoadFormatXyzw, false, 4, 0x10},
      {false, rdna2::kBufferLoadFormatD16Xyz, true, 2, 0x00320010},
      {true, rdna2::kTbufferLoadFormatX, false, 1, 0x10},
      {true, rdna2::kTbufferLoadFormatD16Xyz, true, 2, 0x00320010},
  };
  amdgpu::GlobalMemPipeline global(&cu_->l1_vector(), &cache_);
  constexpr uint32_t sentinel = 0xfeedbeef;
  constexpr uint32_t format = 60; // GFX10 8_8_8_8_UINT.
  for (unsigned word = 0; word < 4; ++word)
    memory_.write32(0x4000 + 4 * word, 0x76543210);
  write_scalar(0, 0x4000);
  write_scalar(1, 0);
  write_scalar(2, 256);
  write_scalar(3,
               (3u << 28) | (1u << 24) | (format << 12) | 4u | (5u << 3) | (6u << 6) | (7u << 9));
  for (const auto &load : loads) {
    // In each form, place the last data/status VGPR at v15 or v16.
    for (uint8_t tfe : {0, 1}) {
      for (unsigned overflow : {0u, 1u}) {
        const auto first = static_cast<uint8_t>(16 - load.data_registers - tfe + overflow);
        SCOPED_TRACE(testing::Message() << "typed=" << load.typed << " op=" << load.opcode
                                        << " D16=" << load.packed_d16 << " TFE=" << unsigned(tfe)
                                        << " first=" << unsigned(first));
        for (unsigned reg = first; reg < 16; ++reg)
          write_vector(reg, sentinel);
        const auto words =
            load.typed ? rdna2::build_mtbuf(load.opcode, {.format = format,
                                                          .vdata = first,
                                                          .opm = load.packed_d16,
                                                          .tfe = tfe,
                                                          .soffset = 128})
                       : rdna2::build_mubuf(
                             load.opcode,
                             {.opm = load.packed_d16, .vdata = first, .tfe = tfe, .soffset = 128});
        auto inst = execute(words);
        ASSERT_TRUE(inst && inst->data());
        auto *state = inst->data_as<amdgpu::VectorMemState>();
        EXPECT_EQ(state->exec_mask, overflow ? 0u : 1u);
        EXPECT_EQ(state->lane_mask, overflow ? 0u : 1u);
        const auto obligations = inst->amdgpu_memory_issue_info()->counter_obligations();
        ASSERT_EQ(obligations.size(), 1u);
        EXPECT_EQ(obligations[0].wait_counter_type(), amdgpu::WaitCounterType::VMCNT);
        ASSERT_EQ(global.issue(inst.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(wave_->exec(), 1u);
        if (overflow) {
          for (unsigned reg = first; reg < 16; ++reg)
            EXPECT_EQ(read_vector(reg), sentinel);
        } else {
          EXPECT_EQ(read_vector(first), load.first_value);
          if (tfe) {
            EXPECT_EQ(read_vector(first + load.data_registers), sentinel);
          }
        }
      }
    }
  }
}

TEST_F(Rdna2MemoryTest, DsAndGdsNullifyTheWholeReturnBeforeChangingStorage) {
  amdgpu::LocalMemPipeline local;
  wave_->set_m0((128u << 16) | 16);
  write_vector(0, 0);
  write_vector(1, 5);
  write_vector(2, 7);
  for (uint8_t indexed_gds : {0, 1}) {
    SCOPED_TRACE(unsigned(indexed_gds));
    const auto backing = [&](auto action) {
      if (indexed_gds)
        gds_->access_gds(wave_->cu().arch(), 0, [&](amdgpu::Lds &b) { action(b, 128u); });
      else
        action(wave_->lds(), 1024u);
    };
    backing([](auto &b, uint32_t base) {
      b.write32(base, 10);
      b.write32(base + 4, 20);
    });
    const auto check = [&](uint16_t op, uint8_t dst, bool active, uint8_t offset1 = 1) {
      auto inst = execute(rdna2::build_ds(op, {.offset1 = offset1,
                                               .gds = indexed_gds,
                                               .addr = 0,
                                               .data0 = 1,
                                               .data1 = 2,
                                               .vdst = dst}));
      ASSERT_TRUE(inst && inst->data());
      auto *d = inst->data_as<amdgpu::VectorMemState>();
      EXPECT_EQ(d->exec_mask, active ? 1u : 0u);
      EXPECT_EQ(d->lane_mask, active ? 1u : 0u);
      ASSERT_EQ(local.issue(inst.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
    };
    write_vector(15, 0xfeedbeef);
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kDsRead2B32, 15, false));
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kDsWrxchg2RtnB32, 15, false));
    EXPECT_EQ(read_vector(15), 0xfeedbeefu);
    backing([](auto &b, uint32_t base) {
      EXPECT_EQ(b.read32(base), 10u);
      EXPECT_EQ(b.read32(base + 4), 20u);
    });
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kDsWrxchg2RtnB32, 14, true));
    EXPECT_EQ(read_vector(14), 10u);
    EXPECT_EQ(read_vector(15), 20u);
    backing([](auto &b, uint32_t base) {
      EXPECT_EQ(b.read32(base), 5u);
      EXPECT_EQ(b.read32(base + 4), 7u);
    });
    // The unused VDST encoding cannot suppress a no-return atomic.
    auto no_return =
        rdna2::build_ds(rdna2::kDsAddU32, {.gds = indexed_gds, .addr = 0, .data0 = 1, .vdst = 16});
    issue(no_return, local);
    backing([](auto &b, uint32_t base) { EXPECT_EQ(b.read32(base), 10u); });
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kDsReadB32, 16, false, 0));
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kDsAddRtnU32, 16, false, 0));
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kDsReadB32, 8, true, 0));
    EXPECT_EQ(read_vector(8), 10u);
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kDsAddRtnU32, 8, true, 0));
    EXPECT_EQ(read_vector(8), 10u);
    backing([](auto &b, uint32_t base) { EXPECT_EQ(b.read32(base), 15u); });
  }
}

TEST_F(Rdna2MemoryTest, FlatAndBufferAtomicsCheckTheReturnedWidthBeforeRmw) {
  amdgpu::GlobalMemPipeline global(&cu_->l1_vector(), &cache_);
  write_scalar(0, 0x4000);
  write_scalar(1, 0);
  write_scalar(2, 256);
  write_scalar(3, (3u << 28) | (1u << 24) | (20u << 12));
  // Exercise both FLAT aperture routing and explicit GLOBAL addresses.
  for (uint8_t segment : {0, 2}) {
    SCOPED_TRACE(unsigned(segment));
    write_vector(0, segment == 0 ? 0x4000u : 0u);
    write_vector(1, 0);
    write_vector(2, 5);
    write_vector(3, 0);
    memory_.write32(0x4000, 10);
    const auto check = [&](uint16_t opcode, uint8_t dst, uint8_t glc, bool active) {
      auto inst =
          execute(rdna2::build_flat(opcode, {.seg = segment,
                                             .glc = glc,
                                             .addr = 0,
                                             .data = 2,
                                             .saddr = static_cast<uint8_t>(segment == 0 ? 125 : 0),
                                             .vdst = dst}));
      ASSERT_TRUE(inst && inst->data());
      EXPECT_EQ(inst->data_as<amdgpu::VectorMemState>()->lane_mask, active ? 1u : 0u);
      ASSERT_EQ(global.issue(inst.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
    };
    write_vector(15, 0xfeedbeef);
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kFlatLoadDwordx2, 15, 0, false));
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kFlatAtomicAddX2, 15, 1, false));
    EXPECT_EQ(read_vector(15), 0xfeedbeefu);
    EXPECT_EQ(memory_.read32(0x4000), 10u);
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kFlatAtomicAdd, 15, 1, true));
    EXPECT_EQ(read_vector(15), 10u);
    ASSERT_NO_FATAL_FAILURE(check(rdna2::kFlatAtomicAdd, 16, 0, true));
    EXPECT_EQ(memory_.read32(0x4000), 20u);
  }
  auto scratch =
      execute(rdna2::build_flat(rdna2::kFlatLoadDwordx2, {.seg = 1, .saddr = 125, .vdst = 15}));
  ASSERT_TRUE(scratch && scratch->data());
  EXPECT_EQ(scratch->data_as<amdgpu::VectorMemState>()->lane_mask, 0u);
  ASSERT_EQ(global.issue(scratch.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
  memory_.write32(0x4000, 10);
  write_vector(15, 5);
  const auto mubuf = [&](uint16_t op, uint8_t reg, uint8_t glc) {
    return rdna2::build_mubuf(op, {.glc = glc, .vdata = reg, .soffset = 128});
  };
  auto invalid = execute(mubuf(rdna2::kBufferAtomicAddX2, 15, 1));
  ASSERT_TRUE(invalid && invalid->data());
  EXPECT_EQ(invalid->data_as<amdgpu::VectorMemState>()->lane_mask, 0u);
  ASSERT_EQ(global.issue(invalid.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(read_vector(15), 5u);
  EXPECT_EQ(memory_.read32(0x4000), 10u);
  issue(mubuf(rdna2::kBufferAtomicAdd, 15, 0), global);
  EXPECT_EQ(read_vector(15), 5u);
  EXPECT_EQ(memory_.read32(0x4000), 15u);
  // CMP_SWAP takes two inputs and returns only one DWORD.
  write_vector(14, 42);
  write_vector(15, 15);
  issue(mubuf(rdna2::kBufferAtomicCmpswap, 14, 1), global);
  EXPECT_EQ(read_vector(14), 15u);
  EXPECT_EQ(read_vector(15), 15u);
  EXPECT_EQ(memory_.read32(0x4000), 42u);
  // LDS destinations use M0 and have no returned VGPR tuple.
  issue(rdna2::build_flat(rdna2::kFlatLoadDword, {.lds = 1, .seg = 2, .saddr = 0, .vdst = 16}),
        global);
  EXPECT_EQ(wave_->lds().read32(1024), 42u);
}

TEST_F(Rdna2MemoryTest, NullifiedMemoryReturnsStillAcquireAndReleaseTheirCounters) {
  test::execution_plugin::CounterObservingPipeline probe;
  const std::array local = rdna2::build_ds(rdna2::kDsAddRtnU32, {.vdst = 16});
  const std::array global =
      rdna2::build_flat(rdna2::kFlatAtomicAdd, {.seg = 2, .glc = 1, .vdst = 16});
  const std::array buffer =
      rdna2::build_mubuf(rdna2::kBufferAtomicAdd, {.glc = 1, .vdata = 16, .soffset = 128});
  for (const auto &words : {local, global, buffer}) {
    auto inst = execute(words);
    ASSERT_TRUE(inst && inst->data());
    EXPECT_EQ(inst->data_as<amdgpu::VectorMemState>()->exec_mask, 0u);
    ASSERT_EQ(probe.issue(inst.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(probe.counters_at_access.lgkmcnt, words == local ? 1u : 0u);
    EXPECT_EQ(probe.counters_at_access.vmcnt, words == local ? 0u : 1u);
    EXPECT_EQ(wave_->wait_counters().lgkmcnt, 0u);
    EXPECT_EQ(wave_->wait_counters().vmcnt, 0u);
    EXPECT_EQ(wave_->exec(), 1u);
  }
}

TEST_F(Rdna2MemoryTest, RegisterLanePermutationsRespectTheLogicalDestination) {
  write_vector(0, 0);
  write_vector(1, 0x12345678);
  for (uint16_t op : {rdna2::kDsSwizzleB32, rdna2::kDsPermuteB32, rdna2::kDsBpermuteB32}) {
    SCOPED_TRACE(op);
    const uint8_t address = op == rdna2::kDsSwizzleB32 ? 1 : 0;
    for (uint8_t destination : {15, 16}) {
      write_vector(destination, 0xabcdef01);
      auto inst = execute(
          rdna2::build_ds(op, {.offset1 = 0x80, .addr = address, .data0 = 1, .vdst = destination}));
      ASSERT_TRUE(inst);
      EXPECT_EQ(inst->data(), nullptr);
      EXPECT_EQ(read_vector(destination), destination == 15 ? 0x12345678u : 0xabcdef01u);
      EXPECT_EQ(wave_->exec(), 1u);
    }
  }
}

TEST_F(Rdna2MemoryTest, InvalidDescriptorAndLdsUploadAreBounded) {
  amdgpu::GlobalMemPipeline global(&cu_->l1_vector(), &cache_);
  memory_.write32(0x4000, 0x76543210);
  write_scalar(0, 0x4000);
  write_scalar(1, 0);
  write_scalar(2, 256);
  write_scalar(3, (3u << 28) | (1u << 24));
  write_vector(8, 1);
  issue(rdna2::build_mubuf(rdna2::kBufferLoadDword, {.vdata = 8, .soffset = 128}), global);
  EXPECT_EQ(read_vector(8), 0u);
  write_scalar(3, (3u << 28) | (1u << 24) | (20u << 12));
  wave_->set_m0(0x10040);
  issue(rdna2::build_mubuf(rdna2::kBufferLoadDword, {.lds = 1, .soffset = 128}), global);
  EXPECT_EQ(wave_->lds().read32(1088), 0x76543210u);
  wave_->set_m0(1024);
  wave_->lds().write32(2048, 0xcccc);
  issue(rdna2::build_mubuf(rdna2::kBufferLoadDword, {.lds = 1, .soffset = 128}), global);
  EXPECT_EQ(wave_->lds().read32(2048), 0xccccu);
}

TEST_F(Rdna2MemoryTest, BufferLdsDestinationIsRestrictedToDocumentedLoadForms) {
  for (auto op : {rdna2::kBufferLoadUbyte, rdna2::kBufferLoadSbyte, rdna2::kBufferLoadUshort,
                  rdna2::kBufferLoadSshort, rdna2::kBufferLoadDword, rdna2::kBufferLoadFormatX}) {
    SCOPED_TRACE(op);
    EXPECT_TRUE(decoder_->decode_window(rdna2::build_mubuf(op, {.lds = 1})).succeeded());
    EXPECT_TRUE(decoder_->decode_window(rdna2::build_mubuf(op, {.lds = 1, .tfe = 1})).failed());
  }
  for (auto op : {rdna2::kBufferLoadDwordx2, rdna2::kBufferLoadFormatXy, rdna2::kBufferStoreDword,
                  rdna2::kBufferAtomicAdd}) {
    SCOPED_TRACE(op);
    EXPECT_TRUE(decoder_->decode_window(rdna2::build_mubuf(op, {.lds = 1})).failed());
  }
  EXPECT_TRUE(
      decoder_
          ->decode_window(rdna2::build_mubuf(rdna2::kBufferLoadFormatD16X, {.lds = 1, .opm = 1}))
          .failed());
}

TEST_F(Rdna2MemoryTest, ScratchBoundsAreCheckedInLogicalLaneBytes) {
  amdgpu::GlobalMemPipeline global(&cu_->l1_vector(), &cache_);
  wave_->set_scratch_base(0x10000);
  wave_->set_scratch_lane_size(8);
  memory_.write32(0x10100, 0xabcd);
  memory_.write32(0x10200, 0xfeedbeef);
  write_vector(0, 4);
  issue(rdna2::build_flat(rdna2::kFlatLoadDwordx2, {.seg = 1, .addr = 0, .saddr = 125, .vdst = 6}),
        global);
  EXPECT_EQ(read_vector(6), 0xabcdu);
  EXPECT_EQ(read_vector(7), 0u);
  write_vector(0, 8);
  issue(rdna2::build_flat(rdna2::kFlatLoadDword, {.seg = 1, .addr = 0, .saddr = 125, .vdst = 6}),
        global);
  EXPECT_EQ(read_vector(6), 0u);
}

TEST_F(Rdna2MemoryTest, FlatWaitRetainsTheFirstCompletePrivateElement) {
  constexpr uint64_t priv = uint64_t{3} << 32;
  cu_->set_apertures(0, 0, priv, priv + UINT32_MAX);
  wave_->set_apertures(0, 0, priv, priv + UINT32_MAX, /*override_launch=*/true);
  wave_->set_scratch_lane_size(2048);
  wave_->set_scratch_base(0x10000);
  wave_->set_exec(7);
  const uint32_t base = wave_->vgpr_alloc().base;
  for (unsigned lane = 0; lane != 3; ++lane) {
    cu_->write_vgpr(base, lane, std::array{0u, 4u, 2u}[lane]);
    cu_->write_vgpr(base + 1, lane, priv >> 32);
  }
  // 0xffc is 2044 in generic GFX10 FLAT. Lane 0 retains one DWORD,
  // lane 1 starts past scratch, and lane 2 has only two bytes remaining.
  auto inst = execute(rdna2::build_flat(rdna2::kFlatLoadDwordx2,
                                        {.offset = 0xffc, .addr = 0, .saddr = 125, .vdst = 8}));
  ASSERT_NE(inst, nullptr);
  const auto plan = amdgpu::MemoryWaitScoreboard::flat_lanes(*inst, *wave_, 0, 0);
  EXPECT_EQ(plan.requests, 1u);
  EXPECT_EQ(plan.shared, 0u);
  const auto *data = inst->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(data->lane_mask, 1u);
  ASSERT_EQ(data->element_lane_masks.size(), 2u);
  EXPECT_EQ(data->element_lane_masks[0], 1u);
  EXPECT_EQ(data->element_lane_masks[1], 0u);
}

TEST_F(Rdna2MemoryTest, AddressedPrivateAndFlatLdsElementsReportAtCompletion) {
  constexpr uint64_t shared = 0x0001'0000'0000'0000, priv = 0x0002'0000'0000'0000;
  constexpr uint32_t sticky = 1u << 3, mem_viol = 1u << 8;
  cu_->set_apertures(shared, shared + 0xffff, priv, priv + 0xffff);
  wave_->set_apertures(shared, shared + 0xffff, priv, priv + 0xffff, /*override_launch=*/true);
  wave_->set_scratch_base(0x10000);
  wave_->set_scratch_lane_size(8);
  wave_->set_lds_size(8);
  for (uint64_t base : {uint64_t{0}, priv, shared}) {
    const bool scratch = base == 0;
    const auto check = [&](uint16_t op, unsigned offset, bool returns, bool expected,
                           unsigned dst = 8, uint64_t exec = 1) {
      SCOPED_TRACE(::testing::Message()
                   << base << ", op " << op << ", offset " << offset << ", returns " << returns
                   << ", dst " << dst << ", exec " << exec);
      wave_->set_exec(exec);
      wave_->set_trapsts(sticky);
      write_vector(0, offset);
      write_vector(1, base >> 32);
      write_vector(2, 1);
      auto inst = execute(rdna2::build_flat(op, {.seg = static_cast<uint8_t>(scratch),
                                                 .glc = static_cast<uint8_t>(returns),
                                                 .addr = 0,
                                                 .data = 2,
                                                 .saddr = 125,
                                                 .vdst = static_cast<uint8_t>(dst)}));
      ASSERT_TRUE(inst && inst->data());
      EXPECT_EQ(wave_->trapsts(), sticky);
      ASSERT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu_, inst.release(), *wave_),
                amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(wave_->trapsts(), sticky | (expected ? mem_viol : 0));
      EXPECT_TRUE(wave_->wait_counters().empty());
    };
    check(rdna2::kFlatLoadDwordx2, 0, false, false);
    check(rdna2::kFlatLoadDwordx2, 4, false, true);
    check(rdna2::kFlatStoreDwordx2, 4, false, true);
    check(rdna2::kFlatLoadDword, 8, false, true);
    check(rdna2::kFlatStoreDword, 8, false, true);
    check(rdna2::kFlatStoreDword, 8, false, false, 8, 0);
    check(rdna2::kFlatLoadDwordx2, 8, false, false, 15);
    if (scratch) {
      // An out-of-range LDS destination masks the lane before it reads scratch.
      const auto load =
          rdna2::build_flat(rdna2::kFlatLoadDword, {.lds = 1, .seg = 1, .addr = 0, .saddr = 125});
      wave_->set_m0(4); // Only lane 0 has a destination in the eight-byte allocation.
      for (unsigned bad_lane : {0u, 1u}) {
        wave_->set_exec(3);
        wave_->set_trapsts(sticky);
        write_vector(0, bad_lane == 0 ? 8 : 0, 0);
        write_vector(0, bad_lane == 1 ? 8 : 0, 1);
        auto inst = execute(load);
        ASSERT_TRUE(inst && inst->data());
        ASSERT_EQ(test::ComputeUnitTestAccess::route_memory_inst(*cu_, inst.release(), *wave_),
                  amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(wave_->trapsts(), sticky | (bad_lane == 0 ? mem_viol : 0));
      }
    }
    if (!scratch) {
      check(rdna2::kFlatAtomicAdd, 8, true, true);
      check(rdna2::kFlatAtomicAdd, 8, false, true);
      check(rdna2::kFlatAtomicAdd, 8, true, false, 16);
      check(rdna2::kFlatAtomicCmpswap, 4, true, false); // One addressed DWORD, two inputs.
    }
  }
}

TEST(Rdna2MemoryDispatchTest, PrivateViolationSurvivesWaitAndClearsThroughTrapsts) {
  constexpr uint64_t priv = 0x0002'0000'0000'0000;
  constexpr uint16_t trapsts_mem_viol = 3 | (8 << 6);
  for (uint32_t width : {32u, 64u}) {
    SCOPED_TRACE(width);
    test::execution_plugin::PluginFixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize(1, "rdna2", width, 106));
    f.cu()->set_apertures(0, 0, priv, priv + 0xffff);
    std::vector<uint32_t> code;
    const auto emit = [&](auto words) { code.insert(code.end(), words.begin(), words.end()); };
    const auto mov = [&](uint8_t reg, uint32_t value) {
      emit(rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = 255, .vdst = reg}));
      code.push_back(value);
    };
    mov(1, priv >> 32);
    mov(2, 0x4000);
    mov(3, 0);
    mov(4, 0x12345678);
    unsigned result = 0;
    const auto access = [&](unsigned offset, bool store) {
      mov(0, offset);
      emit(rdna2::build_flat(store ? rdna2::kFlatStoreDword : rdna2::kFlatLoadDword,
                             {.addr = 0, .data = 4, .saddr = 125, .vdst = 8}));
      code.push_back(0xbf8c0000); // s_waitcnt 0
      emit(rdna2::build_sopk(rdna2::kSWaitcntVscnt, {.sdst = 125}));
      emit(rdna2::build_sopk(rdna2::kSGetregB32, {.simm16 = trapsts_mem_viol, .sdst = 6}));
      emit(rdna2::build_vop1(rdna2::kVMovB32Vop1, {.src0 = 6, .vdst = 9}));
      emit(rdna2::build_flat(rdna2::kFlatStoreDword, {.offset = static_cast<uint16_t>(result++ * 4),
                                                      .seg = 2,
                                                      .addr = 2,
                                                      .data = 9,
                                                      .saddr = 125}));
    };
    access(0, true);
    access(0, false);
    emit(rdna2::build_flat(rdna2::kFlatStoreDword,
                           {.offset = 20, .seg = 2, .addr = 2, .data = 8, .saddr = 125}));
    // Offset 1024 is outside both the four-byte logical limit and the entire rounded wave slice.
    access(1024, false);
    emit(rdna2::build_flat(rdna2::kFlatStoreDword,
                           {.offset = 24, .seg = 2, .addr = 2, .data = 8, .saddr = 125}));
    access(0, false);
    emit(rdna2::build_sopk(rdna2::kSSetregImm32B32, {.simm16 = trapsts_mem_viol}));
    code.push_back(0);
    access(0, false);
    code.push_back(0xbf810000); // s_endpgm
    const auto kernel = f.write_kernel(0x1000, code.data(), code.size());
    using namespace amdhsa;
    f.mem->write32(kernel + offsetof(kernel_descriptor_t, private_segment_fixed_size), 4);
    auto props = f.mem->read32(kernel + offsetof(kernel_descriptor_t, kernel_code_properties));
    RJ_AMDHSA_BITS_SET(props, KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, width == 32);
    f.mem->write16(kernel + offsetof(kernel_descriptor_t, kernel_code_properties), props);
    for (unsigned i = 0; i < 7; ++i)
      f.mem->write32(0x4000 + 4 * i, 0xdeadbeef);
    test::AqlQueue queue(f.mem, f.cp());
    queue.dispatch(kernel, 1, width);
    f.run_until_idle();
    ASSERT_TRUE(f.cu()->is_idle());
    constexpr std::array<unsigned, 7> expected{0, 0, 1, 1, 0, 0x12345678, 0};
    for (unsigned i = 0; i < expected.size(); ++i)
      EXPECT_EQ(f.mem->read32(0x4000 + 4 * i), expected[i]) << "output " << i;
  }
}

TEST_F(Rdna2MemoryTest, IndexedGdsSignalsRejectedAddressesAtCompletion) {
  amdgpu::LocalMemPipeline local;
  constexpr uint32_t base = 128;
  constexpr uint32_t sticky = 1u << 2, mem_viol = 1u << 8;
  wave_->set_m0((base << 16) | 8);
  write_vector(1, 1);
  write_vector(2, 0);
  const auto check = [&](uint16_t op, uint32_t addr, bool fault, uint8_t offset1 = 0,
                         uint8_t dst = 4, uint64_t exec = 1) {
    SCOPED_TRACE(::testing::Message()
                 << op << " address=" << addr << " offset1=" << unsigned(offset1));
    wave_->set_exec(exec);
    wave_->set_trapsts(sticky);
    write_vector(0, addr);
    auto inst = execute(rdna2::build_ds(
        op, {.offset1 = offset1, .gds = 1, .addr = 0, .data0 = 1, .data1 = 2, .vdst = dst}));
    ASSERT_TRUE(inst && inst->data());
    auto &data = *inst->data_as<amdgpu::VectorMemState>();
    if (data.gds_m0)
      amdgpu::LocalMemPipeline::capture_gds_violations(data, *wave_);
    EXPECT_EQ(data.address_range_violation_lanes, fault ? exec : 0u);
    // Check the memory pipeline's completion independently of the routing observation.
    data.address_range_violation_lanes = 0;
    EXPECT_EQ(wave_->trapsts(), sticky);
    ASSERT_EQ(local.issue(inst.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(wave_->trapsts(), sticky | (fault ? mem_viol : 0u));
  };
  gds_->access_gds(wave_->cu().arch(), 0, [](amdgpu::Lds &b) {
    b.write32(base, 17);
    b.write32(base + 4, 23);
    b.write32(base + 8, 99);
  });
  check(rdna2::kDsReadB32, 4, false);
  EXPECT_EQ(read_vector(4), 23u);
  check(rdna2::kDsReadB32, 8, true);
  EXPECT_EQ(read_vector(4), 0u);
  check(rdna2::kDsReadB128, 4, true); // The first DWORD of four still fits.
  EXPECT_EQ(read_vector(4), 23u);
  EXPECT_EQ(read_vector(5), 0u);
  EXPECT_EQ(read_vector(6), 0u);
  EXPECT_EQ(read_vector(7), 0u);
  check(rdna2::kDsReadB64, 4, true); // A single eight-byte element crosses the end.
  EXPECT_EQ(read_vector(4), 0u);
  EXPECT_EQ(read_vector(5), 0u);
  check(rdna2::kDsWriteB32, 8, true);
  check(rdna2::kDsRead2B32, 4, false); // Both offsets address the last DWORD.
  EXPECT_EQ(read_vector(4), 23u);
  EXPECT_EQ(read_vector(5), 23u);
  check(rdna2::kDsRead2B32, 4, true, 1);
  EXPECT_EQ(read_vector(4), 23u);
  EXPECT_EQ(read_vector(5), 0u);
  check(rdna2::kDsWrite2B32, 4, true, 1);
  wave_->set_m0((base << 16) | 16);
  check(rdna2::kDsReadB32, 1, false); // The modeled queue allows unaligned plain accesses.
  check(rdna2::kDsAddRtnU32, 2, true);
  EXPECT_EQ(read_vector(4), 0u);
  check(rdna2::kDsAddRtnU64, 4, true);
  EXPECT_EQ(read_vector(4), 0u);
  EXPECT_EQ(read_vector(5), 0u);
  constexpr uint32_t sentinel = 0xa5a5b6b6;
  write_vector(15, sentinel);
  check(rdna2::kDsRead2B32, 16, false, 1, 15); // Out-of-range address, suppressed return tuple.
  EXPECT_EQ(read_vector(15), sentinel);
  write_vector(4, sentinel);
  check(rdna2::kDsReadB32, 16, false, 0, 4, 0); // Same invalid address with empty EXEC.
  EXPECT_EQ(read_vector(4), sentinel);
  check(rdna2::kDsReadB32, 16, true);
  EXPECT_EQ(read_vector(4), 0u);
  gds_->access_gds(wave_->cu().arch(), 0, [](amdgpu::Lds &b) {
    EXPECT_EQ(b.read32(base), 17u);
    EXPECT_EQ(b.read32(base + 4), 1u);
    EXPECT_EQ(b.read32(base + 8), 99u);
  });
  wave_->set_m0(((base + 2) << 16) | 8);
  check(rdna2::kDsAppend, 0, true); // APPEND uses M0 as the atomic address.
  wave_->set_m0((0xfffcu << 16) | 8);
  check(rdna2::kDsReadB32, 4, true); // M0's window extends past the 64 KiB backing.
}

TEST_F(Rdna2MemoryTest, IndexedGdsUsesDeviceBackingAndSnapshotsTheM0Window) {
  amdgpu::LocalMemPipeline local;
  wave_->set_process_id(7);
  wave_->set_m0((128u << 16) | 4u);
  wave_->lds().write32(1024, 0xcccc);
  write_vector(0, 0);
  write_vector(1, 0x1111);
  auto store = execute(rdna2::build_ds(rdna2::kDsWriteB32, {.gds = 1, .addr = 0, .data0 = 1}));
  ASSERT_TRUE(store && store->data());
  wave_->set_m0(0);
  EXPECT_EQ(local.issue(store.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(wave_->lds().read32(1024), 0xccccu);

  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA2;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 16;
  std::unique_ptr<amdgpu::ComputeUnitCore> peer;
  ASSERT_NO_FATAL_FAILURE(
      test::initialize_compute_unit(peer, "rdna2_peer", config, &memory_, &cache_));
  ASSERT_TRUE(peer->set_gws_device(gds_));
  auto *peer_wave = peer->dispatch_wf(0, 0, 106, 16, 64);
  ASSERT_NE(peer_wave, nullptr);
  peer_wave->set_process_id(7);
  peer_wave->set_exec(1);
  peer_wave->set_m0((128u << 16) | 4u);
  auto add = [&](unsigned addr, unsigned input, unsigned expected) {
    const unsigned vgpr = peer_wave->vgpr_alloc().base;
    peer->write_vgpr(vgpr, 0, addr);
    peer->write_vgpr(vgpr + 1, 0, input);
    auto op = decoder_->decode_window(
        rdna2::build_ds(rdna2::kDsAddRtnU32, {.gds = 1, .addr = 0, .data0 = 1, .vdst = 2}));
    ASSERT_TRUE(op.succeeded());
    ASSERT_TRUE(peer->execute_instruction(op.value().get(), *peer_wave).succeeded());
    EXPECT_EQ(local.issue(op.value().release(), *peer_wave), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(peer->read_vgpr(vgpr + 2, 0), expected);
  };
  add(0, 1, 0x1111);
  add(4, 3, 0); // Outside the four-byte M0 window.
  peer_wave->set_m0((132u << 16) | 4u);
  add(0, 0, 0); // Adjacent backing retained its initial value.
  peer_wave->set_m0((128u << 16) | 4u);
  peer_wave->set_process_id(8);
  add(0, 5, 0);
  peer_wave->set_process_id(7);
  add(0, 0, 0x1112);
  peer_wave->halt();
  wave_->halt();
  gds_->release_process(7);
  gds_->access_gds(wave_->cu().arch(), 7,
                   [](amdgpu::Lds &backing) { EXPECT_EQ(backing.read32(128), 0u); });
}

} // namespace
} // namespace rocjitsu
