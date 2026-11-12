// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
constexpr uint32_t kOffset = 0xff11u;

using Read = std::tuple<uint32_t, uint32_t, uint8_t, uint32_t>;
using ReadCallback = std::tuple<uint32_t, uint64_t, uint8_t>;

class SnapshotObserver final : public ExecutionPlugin {
public:
  explicit SnapshotObserver(const ComputeUnitCore &cu) : ExecutionPlugin("ds_snapshot"), cu_(cu) {}
  std::set<Read> reads;
  std::vector<ReadCallback> callbacks;
  void onAmdgpuReadVgprLanes(const Wavefront *wf, uint32_t reg, uint64_t lanes,
                             uint8_t bytes) override {
    callbacks.emplace_back(reg - wf->vgpr_alloc().base, lanes, bytes);
    while (lanes) {
      const uint32_t lane = std::countr_zero(lanes);
      lanes &= lanes - 1;
      reads.emplace(reg - wf->vgpr_alloc().base, lane, bytes, cu_.read_vgpr_storage(reg, lane));
    }
  }

private:
  const ComputeUnitCore &cu_;
};

struct SnapshotFixture {
  GpuMemory memory{"ds_snapshot_mem"};
  L2Cache l2{"ds_snapshot_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  Wavefront *wf = nullptr;
  rj_code_arch_t arch;
  uint32_t register_count;
  std::shared_ptr<ExecutionPluginGroup> group;
  SnapshotObserver *observer;
  explicit SnapshotFixture(uint32_t wave_size, rj_code_arch_t target = ROCJITSU_CODE_ARCH_CDNA4,
                           uint32_t registers = 512, uint32_t wave_slots = 1)
      : arch(target), register_count(registers) {
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = wave_slots;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = register_count;
    config.lds_size_kb = 64;
    rocjitsu::test::initialize_compute_unit(cu, "ds_snapshot_cu", config, &memory, &l2);
    if (!cu)
      return;
    wf = cu->dispatch_wf(0, 0, 104, register_count, wave_size);
    group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<SnapshotObserver>(*cu);
    observer = plugin.get();
    group->add(std::move(plugin));
  }
  static uint32_t word(uint32_t reg, uint32_t lane) {
    return (reg * 0x170109u) ^ (lane * 0x40203u);
  }
  void seed() {
    for (uint32_t reg = 0; reg < register_count; ++reg)
      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
        cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, word(reg, lane));
  }
  std::unique_ptr<Instruction> issue(uint16_t opcode, uint8_t data, uint8_t address, bool acc) {
    const auto words = cdna4::build_ds(opcode, {.offset0 = kOffset & 0xff,
                                                .offset1 = kOffset >> 8,
                                                .acc = acc,
                                                .addr = address,
                                                .data0 = data});
    return issue_encoded(words);
  }
  std::unique_ptr<Instruction> issue_encoded(const std::array<uint32_t, 2> &words) {
    auto decoder = Decoder::create(arch);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    if (inst) {
      EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    }
    return inst;
  }
};

TEST(DsMemorySnapshotTest, Cdna2DualStoreUsesFirstDataAndClampsToWorkgroup) {
  SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_lds_base(128);
  fx.wf->set_lds_size(512);
  fx.wf->set_m0(UINT32_MAX);
  fx.wf->set_exec(3);
  fx.cu->write_vgpr(0, 0, 508);
  fx.cu->write_vgpr(0, 1, 512);
  for (unsigned lane : {0u, 1u}) {
    fx.cu->write_vgpr(1, lane, 0x11111111);
    fx.cu->write_vgpr(2, lane, 0x22222222);
  }
  fx.wf->lds().write32(640, 0xcccccccc);
  auto inst =
      fx.issue_encoded(cdna2::build_ds(cdna2::kDsWrite2B32, {.addr = 0, .data0 = 1, .data1 = 2}));
  ASSERT_NE(inst, nullptr);
  amdgpu::LocalMemPipeline pipeline;
  ASSERT_EQ(pipeline.issue(inst.release(), *fx.wf), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(fx.wf->lds().read32(636), 0x11111111u);
  EXPECT_EQ(fx.wf->lds().read32(640), 0xccccccccu);
}
// Use each target's generated field layout and opcodes, including the renamed
// VDS stores, so the coverage checks decoder dispatch as well as the payload.
template <typename Fields, auto Build>
std::array<uint32_t, 2> encode_store(uint16_t opcode, uint8_t data, uint8_t address) {
  Fields fields{};
  fields.offset0 = kOffset & 0xff;
  fields.offset1 = kOffset >> 8;
  fields.addr = address;
  fields.data0 = data;
  return Build(opcode, fields);
}

struct StoreTarget {
  rj_code_arch_t arch;
  std::array<uint32_t, 2> wave_sizes;
  std::array<uint16_t, 4> opcodes;
  std::array<uint32_t, 2> (*encode)(uint16_t, uint8_t, uint8_t);
  std::optional<uint16_t> addtid_opcode;
};

constexpr std::array kStoreTargets = {
    StoreTarget{
        ROCJITSU_CODE_ARCH_CDNA1,
        {64, 0},
        {cdna1::kDsWriteB32Ds, cdna1::kDsWriteB64Ds, cdna1::kDsWriteB96Ds, cdna1::kDsWriteB128Ds},
        encode_store<cdna1::DsBuilderFields, cdna1::build_ds>,
        cdna1::kDsWriteAddtidB32Ds},
    StoreTarget{
        ROCJITSU_CODE_ARCH_CDNA2,
        {64, 0},
        {cdna2::kDsWriteB32Ds, cdna2::kDsWriteB64Ds, cdna2::kDsWriteB96Ds, cdna2::kDsWriteB128Ds},
        encode_store<cdna2::DsBuilderFields, cdna2::build_ds>,
        cdna2::kDsWriteAddtidB32Ds},
    StoreTarget{
        ROCJITSU_CODE_ARCH_CDNA3,
        {64, 0},
        {cdna3::kDsWriteB32Ds, cdna3::kDsWriteB64Ds, cdna3::kDsWriteB96Ds, cdna3::kDsWriteB128Ds},
        encode_store<cdna3::DsBuilderFields, cdna3::build_ds>,
        cdna3::kDsWriteAddtidB32Ds},
    StoreTarget{
        ROCJITSU_CODE_ARCH_CDNA4,
        {64, 0},
        {cdna4::kDsWriteB32Ds, cdna4::kDsWriteB64Ds, cdna4::kDsWriteB96Ds, cdna4::kDsWriteB128Ds},
        encode_store<cdna4::DsBuilderFields, cdna4::build_ds>,
        cdna4::kDsWriteAddtidB32Ds},
    StoreTarget{ROCJITSU_CODE_ARCH_CDNA5,
                {32, 0},
                {cdna5::kDsStoreB32Vds, cdna5::kDsStoreB64Vds, cdna5::kDsStoreB96Vds,
                 cdna5::kDsStoreB128Vds},
                encode_store<cdna5::VdsBuilderFields, cdna5::build_vds>,
                std::nullopt},
    StoreTarget{
        ROCJITSU_CODE_ARCH_RDNA1,
        {32, 64},
        {rdna1::kDsWriteB32Ds, rdna1::kDsWriteB64Ds, rdna1::kDsWriteB96Ds, rdna1::kDsWriteB128Ds},
        encode_store<rdna1::DsBuilderFields, rdna1::build_ds>,
        rdna1::kDsWriteAddtidB32Ds},
    StoreTarget{
        ROCJITSU_CODE_ARCH_RDNA2,
        {32, 64},
        {rdna2::kDsWriteB32Ds, rdna2::kDsWriteB64Ds, rdna2::kDsWriteB96Ds, rdna2::kDsWriteB128Ds},
        encode_store<rdna2::DsBuilderFields, rdna2::build_ds>,
        rdna2::kDsWriteAddtidB32Ds},
    StoreTarget{
        ROCJITSU_CODE_ARCH_RDNA3,
        {32, 64},
        {rdna3::kDsStoreB32Ds, rdna3::kDsStoreB64Ds, rdna3::kDsStoreB96Ds, rdna3::kDsStoreB128Ds},
        encode_store<rdna3::DsBuilderFields, rdna3::build_ds>,
        rdna3::kDsStoreAddtidB32Ds},
    StoreTarget{ROCJITSU_CODE_ARCH_RDNA3_5,
                {32, 64},
                {rdna3_5::kDsStoreB32Ds, rdna3_5::kDsStoreB64Ds, rdna3_5::kDsStoreB96Ds,
                 rdna3_5::kDsStoreB128Ds},
                encode_store<rdna3_5::DsBuilderFields, rdna3_5::build_ds>,
                rdna3_5::kDsStoreAddtidB32Ds},
    StoreTarget{ROCJITSU_CODE_ARCH_RDNA4,
                {32, 64},
                {rdna4::kDsStoreB32Vds, rdna4::kDsStoreB64Vds, rdna4::kDsStoreB96Vds,
                 rdna4::kDsStoreB128Vds},
                encode_store<rdna4::VdsBuilderFields, rdna4::build_vds>,
                rdna4::kDsStoreAddtidB32Vds},
};

TEST(DsMemorySnapshot, CdnaWideDsChecksTheWholeTransferAtTheWorkgroupBoundary) {
  constexpr uint32_t lds_base = 512;
  constexpr uint32_t lds_size = 1536; // A whole number of B96 and B128 transfers.
  constexpr uint32_t guard = 0xfeedface;
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4}) {
    SnapshotFixture fx(64, arch, 32);
    ASSERT_NE(fx.wf, nullptr);
    fx.wf->set_lds_base(lds_base);
    fx.wf->set_lds_size(lds_size);
    fx.wf->set_m0(UINT32_MAX);
    fx.wf->set_exec(1);
    const auto base = fx.wf->vgpr_alloc().base;
    auto &lds = fx.wf->lds();
    LocalMemPipeline pipe;
    auto issue = [&](auto words) {
      auto inst = fx.issue_encoded(words);
      ASSERT_NE(inst, nullptr);
      ASSERT_EQ(pipe.issue(inst.release(), *fx.wf), VmAccessOutcome::Complete);
    };
    for (uint32_t width : {3u, 4u}) {
      // The B96 and B128 DS encodings are shared across these four targets.
      const auto write = cdna4::build_ds(width == 3 ? cdna4::kDsWriteB96Ds : cdna4::kDsWriteB128Ds,
                                         {.addr = 0, .data0 = 4});
      const auto read = cdna4::build_ds(width == 3 ? cdna4::kDsReadB96Ds : cdna4::kDsReadB128Ds,
                                        {.addr = 0, .vdst = 8});
      for (uint32_t offset : {lds_size - 4 * width, lds_size - 4}) {
        SCOPED_TRACE(testing::Message()
                     << "arch=" << arch << " dwords=" << width << " offset=" << offset);
        const bool in_bounds = offset + 4 * width <= lds_size;
        fx.cu->write_vgpr(base, 0, offset);
        for (uint32_t i = 0; i < width; ++i) {
          fx.cu->write_vgpr(base + 4 + i, 0, i + 1);
          fx.cu->write_vgpr(base + 8 + i, 0, guard);
          lds.write32(lds_base + offset + 4 * i, guard);
        }
        ASSERT_NO_FATAL_FAILURE(issue(write));
        ASSERT_NO_FATAL_FAILURE(issue(read));
        for (uint32_t i = 0; i < width; ++i) {
          EXPECT_EQ(lds.read32(lds_base + offset + 4 * i), in_bounds ? i + 1 : guard);
          EXPECT_EQ(fx.cu->read_vgpr(base + 8 + i, 0), in_bounds ? i + 1 : 0u);
        }
      }
    }
  }
}

TEST(DsMemorySnapshot, Cdna1ExchangeNullifiesBothAddressesForUnallocatedSecondResult) {
  SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA1, 32);
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(1);
  fx.wf->set_lds_base(512);
  fx.wf->set_lds_size(512);
  const auto base = fx.wf->vgpr_alloc().base;
  fx.cu->write_vgpr(base, 0, 32);
  fx.cu->write_vgpr(base + 1, 0, 16);
  fx.cu->write_vgpr(base + 2, 0, 24);
  auto &lds = fx.wf->lds();
  LocalMemPipeline pipeline;
  for (auto vdst : {30, 31}) {
    lds.write32(544, 100);
    lds.write32(548, 200);
    fx.cu->write_vgpr(base + 31, 0, 300);
    auto inst = fx.issue_encoded(cdna1::build_ds(
        cdna1::kDsWrxchg2RtnB32,
        {.offset1 = 1, .addr = 0, .data0 = 1, .data1 = 2, .vdst = static_cast<uint8_t>(vdst)}));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(pipeline.issue(inst.release(), *fx.wf), VmAccessOutcome::Complete);
    EXPECT_EQ(lds.read32(544), vdst == 30 ? 16u : 100u);
    EXPECT_EQ(lds.read32(548), vdst == 30 ? 24u : 200u);
    EXPECT_EQ(fx.cu->read_vgpr(base + 31, 0), vdst == 30 ? 200u : 300u);
  }
}

TEST(DsMemorySnapshot, Cdna1DualAddressUsesData0ForEqualOffsetsAndRespectsAllocation) {
  SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA1, 32);
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_lds_base(512);
  fx.wf->set_lds_size(512);
  fx.wf->set_exec(1);
  auto &lds = fx.wf->lds();
  const auto base = fx.wf->vgpr_alloc().base;
  fx.cu->write_vgpr(base + 1, 0, 508);
  fx.cu->write_vgpr(base + 2, 0, 11);
  fx.cu->write_vgpr(base + 3, 0, 22);
  LocalMemPipeline pipeline;
  for (unsigned offset : {0u, 1u}) {
    lds.write32(1020, 0);
    lds.write32(1024, 88);
    auto inst = fx.issue_encoded(cdna1::build_ds(
        cdna1::kDsWrite2B32,
        {.offset1 = static_cast<uint8_t>(offset), .addr = 1, .data0 = 2, .data1 = 3}));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(pipeline.issue(inst.release(), *fx.wf), VmAccessOutcome::Complete);
    EXPECT_EQ(lds.read32(1020), 11u);
    EXPECT_EQ(lds.read32(1024), 88u);
  }
}

TEST(DsMemorySnapshot, Cdna4DualOffsetsRespectAllocationAndCoincidentAddresses) {
  SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA4, 32);
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(1);
  fx.wf->set_lds_base(512);
  fx.wf->set_lds_size(512);
  fx.wf->set_m0(UINT32_MAX);
  const auto base = fx.wf->vgpr_alloc().base;
  fx.cu->write_vgpr(base, 0, 508);
  fx.cu->write_vgpr(base + 1, 0, 17);
  fx.cu->write_vgpr(base + 2, 0, 29);
  LocalMemPipeline pipe;
  auto write = [&](uint8_t first, uint8_t second) {
    auto inst = fx.issue_encoded(
        cdna4::build_ds(cdna4::kDsWrite2B32,
                        {.offset0 = first, .offset1 = second, .addr = 0, .data0 = 1, .data1 = 2}));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(pipe.issue(inst.release(), *fx.wf), VmAccessOutcome::Complete);
  };
  fx.wf->lds().write32(1024, 99);
  ASSERT_NO_FATAL_FAILURE(write(0, 0));
  EXPECT_EQ(fx.wf->lds().read32(1020), 17u);
  ASSERT_NO_FATAL_FAILURE(write(1, 0));
  EXPECT_EQ(fx.wf->lds().read32(1020), 29u);
  EXPECT_EQ(fx.wf->lds().read32(1024), 99u);
  ASSERT_NO_FATAL_FAILURE(write(0, 1));
  EXPECT_EQ(fx.wf->lds().read32(1020), 17u);
  EXPECT_EQ(fx.wf->lds().read32(1024), 99u);
  auto read = fx.issue_encoded(
      cdna4::build_ds(cdna4::kDsRead2B32, {.offset0 = 1, .offset1 = 0, .addr = 0, .vdst = 4}));
  ASSERT_NE(read, nullptr);
  ASSERT_EQ(pipe.issue(read.release(), *fx.wf), VmAccessOutcome::Complete);
  EXPECT_EQ(fx.cu->read_vgpr(base + 4, 0), 0u);
  EXPECT_EQ(fx.cu->read_vgpr(base + 5, 0), 17u);
  auto exchange = fx.issue_encoded(
      cdna4::build_ds(cdna4::kDsWrxchg2RtnB32, {.addr = 0, .data0 = 2, .data1 = 1, .vdst = 4}));
  ASSERT_NE(exchange, nullptr);
  ASSERT_EQ(pipe.issue(exchange.release(), *fx.wf), VmAccessOutcome::Complete);
  EXPECT_EQ(fx.wf->lds().read32(1020), 29u);
  EXPECT_EQ(fx.cu->read_vgpr(base + 4, 0), 17u);
  EXPECT_EQ(fx.cu->read_vgpr(base + 5, 0), 17u);
}

TEST(DsMemorySnapshot, DsOffsetsWrapBeforeRelocation) {
  SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA4, 32);
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(1);
  fx.wf->set_lds_base(512);
  fx.wf->set_lds_size(512);
  fx.wf->set_m0(UINT32_MAX);
  const auto base = fx.wf->vgpr_alloc().base;
  fx.cu->write_vgpr(base, 0, 0xfffffffcu);
  fx.cu->write_vgpr(base + 1, 0, 17);
  fx.cu->write_vgpr(base + 2, 0, 29);
  auto &lds = fx.wf->lds();
  lds.write32(508, 99);
  lds.write32(512 + 252, 73);
  LocalMemPipeline pipe;
  auto issue = [&](auto words) {
    auto inst = fx.issue_encoded(words);
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(pipe.issue(inst.release(), *fx.wf), VmAccessOutcome::Complete);
  };

  ASSERT_NO_FATAL_FAILURE(issue(cdna4::build_ds(cdna4::kDsWriteB32, {.offset0 = 4, .data0 = 2})));
  EXPECT_EQ(lds.read32(512), 29u);
  ASSERT_NO_FATAL_FAILURE(
      issue(cdna4::build_ds(cdna4::kDsWrite2B32, {.offset1 = 1, .data0 = 2, .data1 = 1})));
  EXPECT_EQ(lds.read32(508), 99u);
  EXPECT_EQ(lds.read32(512), 17u);
  ASSERT_NO_FATAL_FAILURE(
      issue(cdna4::build_ds(cdna4::kDsRead2st64B32, {.offset1 = 1, .vdst = 4})));
  EXPECT_EQ(fx.cu->read_vgpr(base + 4, 0), 0u);
  EXPECT_EQ(fx.cu->read_vgpr(base + 5, 0), 73u);
  ASSERT_NO_FATAL_FAILURE(issue(cdna4::build_ds(
      cdna4::kDsWrxchg2RtnB32, {.offset0 = 1, .offset1 = 2, .data0 = 2, .data1 = 1, .vdst = 4})));
  EXPECT_EQ(fx.cu->read_vgpr(base + 4, 0), 17u);
  EXPECT_EQ(lds.read32(512), 29u);
  EXPECT_EQ(lds.read32(516), 17u);
}

TEST(DsMemorySnapshot, Cdna4ReturningAtomicRespectsAccumulatorSplit) {
  SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA4, 64);
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(1);
  fx.wf->set_lds_size(512);
  fx.wf->set_accum_vgpr_base(32);
  const auto base = fx.wf->vgpr_alloc().base;
  fx.cu->write_vgpr(base, 0, 0);
  fx.cu->write_vgpr(base + 1, 0, 29);
  fx.cu->write_vgpr(base + ACC_VGPR_OFFSET + 1, 0, 29);
  for (bool acc : {false, true}) {
    for (uint8_t dest : {31, 32, 64}) {
      SCOPED_TRACE(::testing::Message() << "acc=" << acc << " dest=" << unsigned(dest));
      fx.wf->lds().write32(0, 17);
      const auto result_reg = base + (acc ? ACC_VGPR_OFFSET : 0) + dest;
      fx.cu->write_vgpr(result_reg, 0, 99);
      auto inst = fx.issue_encoded(cdna4::build_ds(
          cdna4::kDsWrxchgRtnB32, {.acc = acc, .addr = 0, .data0 = 1, .vdst = dest}));
      ASSERT_NE(inst, nullptr);
      LocalMemPipeline pipe;
      ASSERT_EQ(pipe.issue(inst.release(), *fx.wf), VmAccessOutcome::Complete);
      EXPECT_EQ(fx.wf->lds().read32(0), dest < 32 ? 29u : 17u);
      EXPECT_EQ(fx.cu->read_vgpr(result_reg, 0), dest < 32 ? 17u : 99u);
    }
  }
}

TEST(DsMemorySnapshot, Cdna4ReturningAtomicChecksEveryDestinationBeforeMemoryUpdate) {
  SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA4, 32);
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(1);
  fx.wf->set_lds_size(512);
  const auto base = fx.wf->vgpr_alloc().base;
  fx.cu->write_vgpr(base, 0, 0);
  fx.cu->write_vgpr(base + 1, 0, 17);
  fx.cu->write_vgpr(base + 2, 0, 29);
  fx.cu->write_vgpr(base + 31, 0, 123);
  fx.wf->lds().write32(0, 41);
  fx.wf->lds().write32(4, 43);
  auto inst = fx.issue_encoded(cdna4::build_ds(
      cdna4::kDsWrxchg2RtnB32, {.offset1 = 1, .addr = 0, .data0 = 1, .data1 = 2, .vdst = 31}));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->data_as<VectorMemState>()->exec_mask, 0u);
  LocalMemPipeline pipe;
  ASSERT_EQ(pipe.issue(inst.release(), *fx.wf), VmAccessOutcome::Complete);
  EXPECT_EQ(fx.wf->lds().read32(0), 41u);
  EXPECT_EQ(fx.wf->lds().read32(4), 43u);
  EXPECT_EQ(fx.cu->read_vgpr(base + 31, 0), 123u);
}

TEST(DsMemorySnapshot, Cdna4BufferInvalidateScopesMatchL2Topology) {
  for (bool multiple_l2 : {false, true}) {
    SCOPED_TRACE(multiple_l2);
    SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA4, 32);
    ASSERT_NE(fx.wf, nullptr);
    fx.wf->set_exec(1);
    fx.l2.set_backing_memory(&fx.memory);
    std::optional<L2Cache> peer;
    if (multiple_l2) {
      peer.emplace("peer_xcd_l2", fx.l2.coherence_domain());
      peer->set_backing_memory(&fx.memory);
    }
    GlobalMemPipeline pipe(&fx.cu->l1_vector(), &fx.l2);
    const auto base = fx.wf->vgpr_alloc().base;
    auto load = [&]() {
      auto inst = fx.issue_encoded(
          cdna4::build_flat(cdna4::kFlatLoadDword, {.seg = 2, .addr = 0, .saddr = 127, .vdst = 2}));
      ASSERT_NE(inst, nullptr);
      ASSERT_EQ(pipe.issue(inst.release(), *fx.wf), VmAccessOutcome::Complete);
    };
    for (uint8_t scope = 0; scope < 4; ++scope) {
      SCOPED_TRACE(scope);
      const uint32_t address = 0x4000 + scope * L2Cache::LINE_SIZE;
      fx.cu->write_vgpr(base, 0, address);
      fx.cu->write_vgpr(base + 1, 0, 0u);
      fx.memory.write32(address, 41);
      ASSERT_NO_FATAL_FAILURE(load());
      EXPECT_EQ(fx.cu->read_vgpr(base + 2, 0), 41u);
      // Distinct contents expose which of the two cache levels was invalidated.
      const uint32_t l2_value = 42;
      ASSERT_EQ(
          fx.l2.write(address, reinterpret_cast<const uint8_t *>(&l2_value), sizeof(l2_value)),
          VmAccessOutcome::Complete);
      fx.memory.write32(address, 43);

      uint32_t coherent_value;
      const uint32_t coherent_address = address + 0x1000;
      fx.memory.write32(coherent_address, 7);
      ASSERT_EQ(fx.l2.read(coherent_address, reinterpret_cast<uint8_t *>(&coherent_value),
                           sizeof(coherent_value), Mtype::CC),
                VmAccessOutcome::Complete);
      fx.memory.write32(coherent_address, 8);
      const auto inst =
          cdna4::build_mubuf(cdna4::kBufferInv, {.sc0 = static_cast<uint8_t>(scope & 1),
                                                 .sc1 = static_cast<uint8_t>(scope >> 1)});
      ASSERT_NE(fx.issue_encoded(inst), nullptr);
      ASSERT_NO_FATAL_FAILURE(load());
      const uint32_t expected = scope < 2 ? 41u : scope == 2 && !multiple_l2 ? 42u : 43u;
      EXPECT_EQ(fx.cu->read_vgpr(base + 2, 0), expected);
      ASSERT_EQ(fx.l2.read(coherent_address, reinterpret_cast<uint8_t *>(&coherent_value),
                           sizeof(coherent_value), Mtype::RW),
                VmAccessOutcome::Complete);
      EXPECT_EQ(coherent_value, 7u);
    }
  }
}

TEST(DsMemorySnapshot, RejectsUnimplementedGdsWithoutPreparingPayload) {
  SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA4, 16);
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(1);
  fx.cu->write_vgpr(fx.wf->vgpr_alloc().base, 0, 0);
  fx.cu->write_vgpr(fx.wf->vgpr_alloc().base + 1, 0, 42);
  fx.cu->set_plugin_group(fx.group);

  cdna4::DsBuilderFields fields{.gds = 1, .addr = 0, .data0 = 1};
  const auto words = cdna4::build_ds(cdna4::kDsWriteB32Ds, fields);
  auto decoder = Decoder::create(fx.arch);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  // The CU translates legacy throws into failures, so also check the callback directly.
  ASSERT_NO_THROW(inst->execute(*inst, fx.wf));
  EXPECT_TRUE(fx.cu->execute_instruction(inst.get(), *fx.wf).failed());
  EXPECT_EQ(fx.wf->instruction_execution_error(),
            InstructionExecutionError::UnimplementedInstruction);
  EXPECT_EQ(inst->data(), nullptr);
  EXPECT_TRUE(fx.observer->reads.empty());

  fields.gds = 0;
  auto supported = fx.issue_encoded(cdna4::build_ds(cdna4::kDsWriteB32Ds, fields));
  ASSERT_NE(supported, nullptr);
  EXPECT_EQ(fx.wf->instruction_execution_error(), InstructionExecutionError::None);
  EXPECT_NE(supported->data(), nullptr);
  fx.wf->halt();
}

TEST(DsMemorySnapshot, DwordStoresPreservePayloadAndObservedValues) {
  constexpr std::array opcodes = {cdna4::kDsWriteB32Ds, cdna4::kDsWriteB64Ds, cdna4::kDsWriteB96Ds,
                                  cdna4::kDsWriteB128Ds};
  constexpr uint32_t wave_size = 64;
  SnapshotFixture fx(wave_size);
  ASSERT_NE(fx.wf, nullptr);
  fx.seed();
  fx.cu->set_plugin_group(fx.group);
  for (uint32_t width = 1; width <= opcodes.size(); ++width)
    for (uint8_t data : {uint8_t{0}, uint8_t{254}, uint8_t{255}})
      for (bool acc : {false, true})
        for (bool alias : {false, true})
          for (uint64_t exec : {uint64_t{0}, uint64_t{0xa55aa55aa55aa55a}, ~uint64_t{0}}) {
            SCOPED_TRACE(testing::Message()
                         << "wave=" << wave_size << " width=" << width << " data=" << unsigned(data)
                         << " acc=" << acc << " alias=" << alias << " exec=" << exec);
            fx.wf->set_exec(exec);
            fx.wf->set_vgpr_write_mask(0x5555555555555555ull);
            fx.observer->reads.clear();
            fx.observer->callbacks.clear();
            const uint8_t address = alias ? data : 7;
            auto inst = fx.issue(opcodes[width - 1], data, address, acc);
            ASSERT_NE(inst, nullptr);
            const auto *state = inst->data_as<VectorMemState>();
            ASSERT_NE(state, nullptr);
            ASSERT_EQ(state->store_data.size(), wave_size * width * sizeof(uint32_t));
            EXPECT_EQ(state->exec_mask, exec);
            EXPECT_EQ(state->lane_mask, exec);
            std::set<Read> expected_reads;
            for (uint32_t lane = 0; lane < wave_size; ++lane) {
              const bool active = (exec >> lane) & 1;
              if (active)
                expected_reads.emplace(address, lane, ExecutionPlugin::kFullByteMask,
                                       SnapshotFixture::word(address, lane));
              const uint32_t expected_address =
                  active ? SnapshotFixture::word(address, lane) + kOffset + fx.wf->lds_base() : 0u;
              EXPECT_EQ(state->per_lane_addr[lane], expected_address);
              for (uint32_t element = 0; element < width; ++element) {
                const uint32_t reg = data + (acc ? 256u : 0u) + element;
                const uint32_t expected =
                    active && reg < 512 ? SnapshotFixture::word(reg, lane) : 0;
                uint32_t actual;
                std::memcpy(&actual, state->store_data.data() + (lane * width + element) * 4, 4);
                EXPECT_EQ(actual, expected);
                if (active && reg < 512)
                  expected_reads.emplace(reg, lane, ExecutionPlugin::kFullByteMask, expected);
              }
            }
            EXPECT_EQ(fx.observer->reads, expected_reads);
            std::vector<ReadCallback> expected_callbacks;
            if (exec) {
              expected_callbacks.emplace_back(address, exec, ExecutionPlugin::kFullByteMask);
              const uint32_t data_base = data + (acc ? 256u : 0u);
              if (data_base + width <= fx.register_count) {
                for (uint32_t element = 0; element < width; ++element)
                  expected_callbacks.emplace_back(data_base + element, exec,
                                                  ExecutionPlugin::kFullByteMask);
              } else {
                for (uint32_t lane = 0; lane < wave_size; ++lane)
                  if ((exec >> lane) & 1)
                    for (uint32_t element = 0; element < width; ++element)
                      if (data_base + element < fx.register_count)
                        expected_callbacks.emplace_back(data_base + element, uint64_t{1} << lane,
                                                        ExecutionPlugin::kFullByteMask);
              }
            }
            EXPECT_EQ(fx.observer->callbacks, expected_callbacks);
          }
}

TEST(DsMemorySnapshot, DualLoadsPreserveAddressesAndPerLaneObservations) {
  for (const auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    const uint32_t wave_size = arch == ROCJITSU_CODE_ARCH_CDNA4 ? 64 : 32;
    SnapshotFixture fx(wave_size, arch, 16);
    ASSERT_NE(fx.wf, nullptr);
    fx.seed();
    const std::array<uint16_t, 4> opcodes =
        arch == ROCJITSU_CODE_ARCH_CDNA4
            ? std::array{cdna4::kDsRead2B32Ds, cdna4::kDsRead2st64B32Ds, cdna4::kDsRead2B64Ds,
                         cdna4::kDsRead2st64B64Ds}
            : std::array{cdna5::kDsLoad2addrB32Vds, cdna5::kDsLoad2addrStride64B32Vds,
                         cdna5::kDsLoad2addrB64Vds, cdna5::kDsLoad2addrStride64B64Vds};
    const uint64_t wave_mask = wave_size == 64 ? ~uint64_t{0} : 0xffffffffull;
    for (bool observed : {false, true}) {
      fx.cu->set_plugin_group(observed ? fx.group : nullptr);
      for (uint32_t kind = 0; kind < opcodes.size(); ++kind)
        for (uint8_t address : {uint8_t{3}, uint8_t{15}, uint8_t{16}})
          for (uint64_t exec : {uint64_t{0}, uint64_t{0xa55aa55aa55aa55a} & wave_mask, wave_mask}) {
            SCOPED_TRACE(testing::Message()
                         << "arch=" << arch << " kind=" << kind << " address=" << unsigned(address)
                         << " observed=" << observed << " exec=" << exec);
            fx.wf->set_exec(exec);
            fx.observer->callbacks.clear();
            std::array<uint32_t, 2> words;
            if (arch == ROCJITSU_CODE_ARCH_CDNA4)
              words = cdna4::build_ds(opcodes[kind],
                                      {.offset0 = 1, .offset1 = 3, .addr = address, .vdst = 4});
            else
              words = cdna5::build_vds(opcodes[kind],
                                       {.offset0 = 1, .offset1 = 3, .addr = address, .vdst = 4});
            auto inst = fx.issue_encoded(words);
            ASSERT_NE(inst, nullptr);
            const auto *state = inst->data_as<VectorMemState>();
            ASSERT_NE(state, nullptr);
            EXPECT_EQ(state->lane_mask, exec);
            EXPECT_EQ(state->exec_mask, exec);
            const uint32_t scale = (kind < 2 ? 4u : 8u) * (kind % 2 ? 64u : 1u);
            std::vector<ReadCallback> expected_callbacks;
            for (uint32_t lane = 0; lane < wave_size; ++lane) {
              const bool active = (exec >> lane) & 1;
              const uint32_t base = address < 16 ? SnapshotFixture::word(address, lane) : 0;
              EXPECT_EQ(state->per_lane_addr[lane], active ? base + scale + fx.wf->lds_base() : 0);
              EXPECT_EQ(state->ds2_per_lane_addr[lane],
                        active ? base + 3 * scale + fx.wf->lds_base() : 0);
              // CDNA4 owns a full physical block including its accumulator
              // bank; v16 is an unseeded zero register there. CDNA5's block
              // ends at v15 in this fixture.
              if (observed && active && (arch == ROCJITSU_CODE_ARCH_CDNA4 || address < 16))
                expected_callbacks.emplace_back(address, uint64_t{1} << lane,
                                                ExecutionPlugin::kFullByteMask);
            }
            EXPECT_EQ(fx.observer->callbacks, expected_callbacks);
          }
    }
  }
}

TEST(DsMemorySnapshot, OutstandingLogicalZeroPayloadSurvivesLaterIssue) {
  for (const auto &target : kStoreTargets) {
    for (uint32_t wave_size : target.wave_sizes) {
      if (wave_size == 0)
        continue;
      SCOPED_TRACE(testing::Message() << "arch=" << target.arch << " wave=" << wave_size);
      SnapshotFixture fx(wave_size, target.arch, 256);
      ASSERT_NE(fx.wf, nullptr);
      fx.wf->set_exec(wave_size == 64 ? ~uint64_t{0} : 0xffffffffu);
      const auto words = target.encode(target.opcodes[1], 0, 0);
      auto first = fx.issue_encoded(words);
      ASSERT_NE(first, nullptr);
      const auto payload = first->data_as<VectorMemState>()->store_data;
      const auto addresses = first->data_as<VectorMemState>()->per_lane_addr;
      EXPECT_TRUE(std::ranges::all_of(payload, [](uint8_t byte) { return byte == 0; }));
      fx.seed();
      auto second = fx.issue_encoded(words);
      ASSERT_NE(second, nullptr);
      EXPECT_NE(second->data_as<VectorMemState>()->store_data, payload);
      EXPECT_EQ(first->data_as<VectorMemState>()->store_data, payload);
      EXPECT_EQ(first->data_as<VectorMemState>()->per_lane_addr, addresses);
    }
  }
}

TEST(DsMemorySnapshot, DwordStoresCoverEveryGeneratedTarget) {
  constexpr uint8_t data = 11, address = 7;
  for (const auto &target : kStoreTargets) {
    for (uint32_t wave_size : target.wave_sizes) {
      if (wave_size == 0)
        continue;
      SnapshotFixture fx(wave_size, target.arch, 256);
      ASSERT_NE(fx.wf, nullptr);
      fx.seed();
      fx.cu->set_plugin_group(fx.group);
      for (size_t store = 0; store <= target.opcodes.size(); ++store) {
        const bool addtid = store == target.opcodes.size();
        // CDNA5 ADDTID uses a separate generator unchanged by this optimization.
        if (addtid && !target.addtid_opcode)
          continue;
        const uint32_t width = addtid ? 1 : store + 1;
        const uint16_t opcode = addtid ? *target.addtid_opcode : target.opcodes[store];
        for (uint64_t exec : {uint64_t{0}, uint64_t{0xa55aa55aa55aa55a}, ~uint64_t{0}}) {
          exec &= wave_size == 64 ? ~uint64_t{0} : 0xffffffffu;
          SCOPED_TRACE(testing::Message()
                       << "arch=" << target.arch << " wave=" << wave_size << " opcode=" << opcode
                       << " width=" << width << " exec=" << exec);
          fx.wf->set_exec(exec);
          fx.observer->reads.clear();
          fx.observer->callbacks.clear();
          auto inst = fx.issue_encoded(target.encode(opcode, data, address));
          ASSERT_NE(inst, nullptr);
          const auto *state = inst->data_as<VectorMemState>();
          ASSERT_NE(state, nullptr);
          ASSERT_EQ(state->store_data.size(), wave_size * width * sizeof(uint32_t));
          EXPECT_EQ(state->exec_mask, exec);
          EXPECT_EQ(state->lane_mask, exec);
          std::set<Read> expected_reads;
          for (uint32_t lane = 0; lane < wave_size; ++lane) {
            const bool active = (exec >> lane) & 1;
            if (active && !addtid)
              expected_reads.emplace(address, lane, ExecutionPlugin::kFullByteMask,
                                     SnapshotFixture::word(address, lane));
            if (!addtid) {
              EXPECT_EQ(state->per_lane_addr[lane],
                        active ? SnapshotFixture::word(address, lane) + kOffset + fx.wf->lds_base()
                               : 0u);
            }
            for (uint32_t element = 0; element < width; ++element) {
              const uint32_t expected = active ? SnapshotFixture::word(data + element, lane) : 0;
              uint32_t actual;
              std::memcpy(&actual, state->store_data.data() + (lane * width + element) * 4, 4);
              EXPECT_EQ(actual, expected);
              if (active)
                expected_reads.emplace(data + element, lane, ExecutionPlugin::kFullByteMask,
                                       expected);
            }
          }
          if (addtid) {
            // Address generation predates this snapshot optimization. Check only
            // the data-register observations without fixing its ADDR behavior here.
            std::erase_if(fx.observer->reads,
                          [](const Read &read) { return std::get<0>(read) == address; });
            std::erase_if(fx.observer->callbacks, [](const ReadCallback &callback) {
              return std::get<0>(callback) == address;
            });
          }
          EXPECT_EQ(fx.observer->reads, expected_reads);
          std::vector<ReadCallback> expected_callbacks;
          if (exec) {
            if (!addtid)
              expected_callbacks.emplace_back(address, exec, ExecutionPlugin::kFullByteMask);
            for (uint32_t element = 0; element < width; ++element)
              expected_callbacks.emplace_back(data + element, exec, ExecutionPlugin::kFullByteMask);
          }
          EXPECT_EQ(fx.observer->callbacks, expected_callbacks);
        }
      }
    }
  }
}

TEST(DsMemorySnapshot, BoundaryFallbackDoesNotReadNeighborWave) {
  for (uint16_t opcode : {cdna4::kDsWriteB96Ds, cdna4::kDsWriteB128Ds}) {
    SnapshotFixture fx(64, ROCJITSU_CODE_ARCH_CDNA4, 512, 2);
    ASSERT_NE(fx.wf, nullptr);
    auto *neighbor = fx.cu->dispatch_wf(1, 0, 104, 512, 64);
    ASSERT_NE(neighbor, nullptr);
    ASSERT_EQ(neighbor->vgpr_alloc().base, fx.wf->vgpr_alloc().base + 512);
    fx.wf->set_exec(1);
    fx.cu->write_vgpr(fx.wf->vgpr_alloc().base + 511, 0, 0x11223344);
    for (uint32_t reg = 0; reg < 3; ++reg)
      fx.cu->write_vgpr(neighbor->vgpr_alloc().base + reg, 0, 0xaabbcc00 + reg);
    fx.cu->set_plugin_group(fx.group);
    auto inst = fx.issue(opcode, 255, 7, true);
    ASSERT_NE(inst, nullptr);
    const auto *state = inst->data_as<VectorMemState>();
    ASSERT_NE(state, nullptr);
    const uint32_t width = opcode == cdna4::kDsWriteB96Ds ? 3 : 4;
    ASSERT_EQ(state->store_data.size(), 64 * width * sizeof(uint32_t));
    for (uint32_t element = 0; element < width; ++element) {
      SCOPED_TRACE(testing::Message() << "width=" << width << " element=" << element);
      uint32_t value;
      std::memcpy(&value, state->store_data.data() + element * sizeof(value), sizeof(value));
      EXPECT_EQ(value, element == 0 ? 0x11223344u : 0u);
    }
    const std::vector<ReadCallback> expected = {{7, 1, ExecutionPlugin::kFullByteMask},
                                                {511, 1, ExecutionPlugin::kFullByteMask}};
    EXPECT_EQ(fx.observer->callbacks, expected);
  }
}

TEST(DsMemorySnapshot, Rdna4Wave32BoundaryFallbackPreservesPayloadAndCallbacks) {
  constexpr uint32_t wave_size = 32;
  constexpr uint8_t data = 255, address = 7;
  constexpr uint64_t exec = (uint64_t{1} << 1) | (uint64_t{1} << 17) | (uint64_t{1} << 31);
  for (uint16_t opcode : {rdna4::kDsStoreB96Vds, rdna4::kDsStoreB128Vds}) {
    SnapshotFixture fx(wave_size, ROCJITSU_CODE_ARCH_RDNA4, 256, 2);
    ASSERT_NE(fx.wf, nullptr);
    auto *neighbor = fx.cu->dispatch_wf(1, 0, 104, 256, wave_size);
    ASSERT_NE(neighbor, nullptr);
    ASSERT_EQ(neighbor->vgpr_alloc().base, fx.wf->vgpr_alloc().base + 256);
    fx.seed();
    for (uint32_t reg = 0; reg < 3; ++reg)
      for (uint32_t lane = 0; lane < wave_size; ++lane)
        fx.cu->write_vgpr(neighbor->vgpr_alloc().base + reg, lane, 0xaabbcc00 + reg);
    fx.wf->set_exec(exec);
    fx.cu->set_plugin_group(fx.group);
    auto inst = fx.issue_encoded(
        encode_store<rdna4::VdsBuilderFields, rdna4::build_vds>(opcode, data, address));
    ASSERT_NE(inst, nullptr);
    const auto *state = inst->data_as<VectorMemState>();
    ASSERT_NE(state, nullptr);
    const uint32_t width = opcode == rdna4::kDsStoreB96Vds ? 3 : 4;
    ASSERT_EQ(state->store_data.size(), wave_size * width * sizeof(uint32_t));
    EXPECT_EQ(state->exec_mask, exec);
    EXPECT_EQ(state->lane_mask, exec);
    std::set<Read> expected_reads;
    std::vector<ReadCallback> expected_callbacks = {
        {address, exec, ExecutionPlugin::kFullByteMask}};
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      SCOPED_TRACE(testing::Message() << "width=" << width << " lane=" << lane);
      const bool active = (exec >> lane) & 1;
      EXPECT_EQ(state->per_lane_addr[lane],
                active ? SnapshotFixture::word(address, lane) + kOffset + fx.wf->lds_base() : 0u);
      for (uint32_t element = 0; element < width; ++element) {
        uint32_t value;
        std::memcpy(&value, state->store_data.data() + (lane * width + element) * sizeof(value),
                    sizeof(value));
        EXPECT_EQ(value, active && element == 0 ? SnapshotFixture::word(data, lane) : 0u);
      }
      if (active) {
        expected_reads.emplace(address, lane, ExecutionPlugin::kFullByteMask,
                               SnapshotFixture::word(address, lane));
        expected_reads.emplace(data, lane, ExecutionPlugin::kFullByteMask,
                               SnapshotFixture::word(data, lane));
        expected_callbacks.emplace_back(data, uint64_t{1} << lane, ExecutionPlugin::kFullByteMask);
      }
    }
    EXPECT_EQ(fx.observer->reads, expected_reads);
    EXPECT_EQ(fx.observer->callbacks, expected_callbacks);
  }
}

} // namespace
