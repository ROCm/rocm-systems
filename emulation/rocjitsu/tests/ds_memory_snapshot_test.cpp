// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
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
  Wavefront *wf;
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
    cu = ComputeUnitCore::create("ds_snapshot_cu", config, &memory, &l2);
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
  uint32_t wave_size;
  std::array<uint16_t, 4> opcodes;
  std::array<uint32_t, 2> (*encode)(uint16_t, uint8_t, uint8_t);
};

constexpr std::array kStoreTargets = {
    StoreTarget{
        ROCJITSU_CODE_ARCH_CDNA1,
        64,
        {cdna1::kDsWriteB32Ds, cdna1::kDsWriteB64Ds, cdna1::kDsWriteB96Ds, cdna1::kDsWriteB128Ds},
        encode_store<cdna1::DsBuilderFields, cdna1::build_ds>},
    StoreTarget{
        ROCJITSU_CODE_ARCH_CDNA2,
        64,
        {cdna2::kDsWriteB32Ds, cdna2::kDsWriteB64Ds, cdna2::kDsWriteB96Ds, cdna2::kDsWriteB128Ds},
        encode_store<cdna2::DsBuilderFields, cdna2::build_ds>},
    StoreTarget{
        ROCJITSU_CODE_ARCH_CDNA3,
        64,
        {cdna3::kDsWriteB32Ds, cdna3::kDsWriteB64Ds, cdna3::kDsWriteB96Ds, cdna3::kDsWriteB128Ds},
        encode_store<cdna3::DsBuilderFields, cdna3::build_ds>},
    StoreTarget{
        ROCJITSU_CODE_ARCH_CDNA4,
        64,
        {cdna4::kDsWriteB32Ds, cdna4::kDsWriteB64Ds, cdna4::kDsWriteB96Ds, cdna4::kDsWriteB128Ds},
        encode_store<cdna4::DsBuilderFields, cdna4::build_ds>},
    StoreTarget{ROCJITSU_CODE_ARCH_CDNA5,
                32,
                {cdna5::kDsStoreB32Vds, cdna5::kDsStoreB64Vds, cdna5::kDsStoreB96Vds,
                 cdna5::kDsStoreB128Vds},
                encode_store<cdna5::VdsBuilderFields, cdna5::build_vds>},
    StoreTarget{
        ROCJITSU_CODE_ARCH_RDNA1,
        32,
        {rdna1::kDsWriteB32Ds, rdna1::kDsWriteB64Ds, rdna1::kDsWriteB96Ds, rdna1::kDsWriteB128Ds},
        encode_store<rdna1::DsBuilderFields, rdna1::build_ds>},
    StoreTarget{
        ROCJITSU_CODE_ARCH_RDNA2,
        32,
        {rdna2::kDsWriteB32Ds, rdna2::kDsWriteB64Ds, rdna2::kDsWriteB96Ds, rdna2::kDsWriteB128Ds},
        encode_store<rdna2::DsBuilderFields, rdna2::build_ds>},
    StoreTarget{
        ROCJITSU_CODE_ARCH_RDNA3,
        32,
        {rdna3::kDsStoreB32Ds, rdna3::kDsStoreB64Ds, rdna3::kDsStoreB96Ds, rdna3::kDsStoreB128Ds},
        encode_store<rdna3::DsBuilderFields, rdna3::build_ds>},
    StoreTarget{ROCJITSU_CODE_ARCH_RDNA3_5,
                32,
                {rdna3_5::kDsStoreB32Ds, rdna3_5::kDsStoreB64Ds, rdna3_5::kDsStoreB96Ds,
                 rdna3_5::kDsStoreB128Ds},
                encode_store<rdna3_5::DsBuilderFields, rdna3_5::build_ds>},
    StoreTarget{ROCJITSU_CODE_ARCH_RDNA4,
                32,
                {rdna4::kDsStoreB32Vds, rdna4::kDsStoreB64Vds, rdna4::kDsStoreB96Vds,
                 rdna4::kDsStoreB128Vds},
                encode_store<rdna4::VdsBuilderFields, rdna4::build_vds>},
};

TEST(DsMemorySnapshot, DwordStoresPreservePayloadAndObservedValues) {
  constexpr std::array opcodes = {cdna4::kDsWriteB32Ds, cdna4::kDsWriteB64Ds, cdna4::kDsWriteB96Ds,
                                  cdna4::kDsWriteB128Ds};
  for (uint32_t wave : {32u, 64u}) {
    SnapshotFixture fx(wave);
    ASSERT_NE(fx.wf, nullptr);
    fx.seed();
    fx.cu->set_plugin_group(fx.group);
    for (uint32_t width = 1; width <= opcodes.size(); ++width)
      for (uint8_t data : {uint8_t{0}, uint8_t{254}, uint8_t{255}})
        for (bool acc : {false, true})
          for (bool alias : {false, true})
            for (uint64_t exec : {uint64_t{0}, uint64_t{0xa55aa55aa55aa55a}, ~uint64_t{0}}) {
              exec &= wave == 64 ? ~uint64_t{0} : 0xffffffffu;
              SCOPED_TRACE(testing::Message()
                           << "wave=" << wave << " width=" << width << " data=" << unsigned(data)
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
              ASSERT_EQ(state->store_data.size(), wave * width * sizeof(uint32_t));
              EXPECT_EQ(state->exec_mask, exec);
              EXPECT_EQ(state->lane_mask, exec);
              std::set<Read> expected_reads;
              for (uint32_t lane = 0; lane < wave; ++lane) {
                const bool active = (exec >> lane) & 1;
                if (active)
                  expected_reads.emplace(address, lane, ExecutionPlugin::kFullByteMask,
                                         SnapshotFixture::word(address, lane));
                const uint32_t expected_address =
                    active ? SnapshotFixture::word(address, lane) + kOffset + fx.wf->lds_base()
                           : 0u;
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
                  for (uint32_t lane = 0; lane < wave; ++lane)
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
}

TEST(DsMemorySnapshot, OutstandingLogicalZeroPayloadSurvivesLaterIssue) {
  for (uint32_t wave : {32u, 64u}) {
    SnapshotFixture fx(wave);
    ASSERT_NE(fx.wf, nullptr);
    fx.wf->set_exec(wave == 64 ? ~uint64_t{0} : 0xffffffffu);
    auto first = fx.issue(cdna4::kDsWriteB64Ds, 0, 0, false);
    ASSERT_NE(first, nullptr);
    const auto payload = first->data_as<VectorMemState>()->store_data;
    const auto addresses = first->data_as<VectorMemState>()->per_lane_addr;
    EXPECT_TRUE(std::ranges::all_of(payload, [](uint8_t byte) { return byte == 0; }));
    fx.seed();
    auto second = fx.issue(cdna4::kDsWriteB64Ds, 0, 0, false);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(second->data_as<VectorMemState>()->store_data, payload);
    EXPECT_EQ(first->data_as<VectorMemState>()->store_data, payload);
    EXPECT_EQ(first->data_as<VectorMemState>()->per_lane_addr, addresses);
  }
}
TEST(DsMemorySnapshot, DwordStoresCoverEveryGeneratedTarget) {
  constexpr uint8_t data = 11, address = 7;
  for (const auto &target : kStoreTargets) {
    SnapshotFixture fx(target.wave_size, target.arch, 256);
    ASSERT_NE(fx.wf, nullptr);
    fx.seed();
    fx.cu->set_plugin_group(fx.group);
    for (uint32_t width = 1; width <= target.opcodes.size(); ++width)
      for (uint64_t exec : {uint64_t{0}, uint64_t{0xa55aa55aa55aa55a}, ~uint64_t{0}}) {
        exec &= target.wave_size == 64 ? ~uint64_t{0} : 0xffffffffu;
        SCOPED_TRACE(testing::Message()
                     << "arch=" << target.arch << " width=" << width << " exec=" << exec);
        fx.wf->set_exec(exec);
        fx.observer->reads.clear();
        fx.observer->callbacks.clear();
        auto inst = fx.issue_encoded(target.encode(target.opcodes[width - 1], data, address));
        ASSERT_NE(inst, nullptr);
        const auto *state = inst->data_as<VectorMemState>();
        ASSERT_NE(state, nullptr);
        ASSERT_EQ(state->store_data.size(), target.wave_size * width * sizeof(uint32_t));
        EXPECT_EQ(state->exec_mask, exec);
        EXPECT_EQ(state->lane_mask, exec);
        std::set<Read> expected_reads;
        for (uint32_t lane = 0; lane < target.wave_size; ++lane) {
          const bool active = (exec >> lane) & 1;
          if (active)
            expected_reads.emplace(address, lane, ExecutionPlugin::kFullByteMask,
                                   SnapshotFixture::word(address, lane));
          EXPECT_EQ(state->per_lane_addr[lane],
                    active ? SnapshotFixture::word(address, lane) + kOffset + fx.wf->lds_base()
                           : 0u);
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
        EXPECT_EQ(fx.observer->reads, expected_reads);
        std::vector<ReadCallback> expected_callbacks;
        if (exec) {
          expected_callbacks.emplace_back(address, exec, ExecutionPlugin::kFullByteMask);
          for (uint32_t element = 0; element < width; ++element)
            expected_callbacks.emplace_back(data + element, exec, ExecutionPlugin::kFullByteMask);
        }
        EXPECT_EQ(fx.observer->callbacks, expected_callbacks);
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
    uint32_t value;
    std::memcpy(&value, state->store_data.data() + 4, 4);
    EXPECT_EQ(value, 0u);
    std::memcpy(&value, state->store_data.data(), 4);
    EXPECT_EQ(value, 0x11223344u);
    const std::vector<ReadCallback> expected = {{7, 1, ExecutionPlugin::kFullByteMask},
                                                {511, 1, ExecutionPlugin::kFullByteMask}};
    EXPECT_EQ(fx.observer->callbacks, expected);
  }
}

} // namespace
