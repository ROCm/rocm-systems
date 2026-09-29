// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/graphics_instructions.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_transfer.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/cpu_dispatch_pool.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/graphics_draw.h"
#include "rocjitsu/vm/amdgpu/graphics_stage.h"
#include "rocjitsu/vm/amdgpu/image_address.h"
#include "rocjitsu/vm/amdgpu/image_bc.h"
#include "rocjitsu/vm/amdgpu/image_filter.h"
#include "rocjitsu/vm/amdgpu/image_metadata.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/raster_math.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/data_types.h"
#include "util/except.h"

#include <barrier>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstring>

#include <gtest/gtest.h>

using namespace rocjitsu;

namespace rocjitsu::amdgpu {
class GraphicsDrawTestAccess {
public:
  static std::vector<uint32_t> fragment_bits(const GraphicsDraw &draw) {
    std::vector<uint32_t> bits{draw.fragment_wave_size_};
    for (const auto &batch : draw.fragments_) {
      bits.push_back(batch.relative_layer);
      bits.push_back(batch.front);
      bits.push_back(batch.parameters.size());
      bits.insert(bits.end(), batch.parameters.begin(), batch.parameters.end());
      for (uint32_t lane = 0; lane < batch.lanes.size(); ++lane) {
        const auto &fragment = batch.lanes[lane];
        bits.push_back(fragment.x);
        bits.push_back(fragment.y);
        bits.push_back(fragment.covered);
        bits.push_back(fragment.stencil_reference);
        bits.push_back(fragment.stencil_exported);
        for (float value :
             {fragment.i, fragment.j, fragment.z, fragment.linear_i, fragment.linear_j,
              fragment.pull_model[0], fragment.pull_model[1], fragment.pull_model[2]})
          bits.push_back(std::bit_cast<uint32_t>(value));
        for (uint32_t target = 0; target < GraphicsDraw::kColorTargets; ++target) {
          const auto &color = draw.fragment_export(batch, lane, target);
          bits.push_back(color.mask);
          bits.insert(bits.end(), color.values.begin(), color.values.end());
        }
      }
    }
    return bits;
  }

  // Used only by the unused-input contract test. Existing complete private
  // snapshots above continue comparing every field without normalization.
  static std::vector<uint32_t> consumed_fragment_bits(const GraphicsDraw &draw) {
    auto consumed = draw;
    if (!(draw.context_[0x198] & 0x70))
      for (auto &batch : consumed.fragments_)
        for (auto &fragment : batch.lanes)
          fragment.linear_i = fragment.linear_j = 0;
    return fragment_bits(consumed);
  }

  static bool linear_values_are_zero(const GraphicsDraw &draw) {
    for (const auto &batch : draw.fragments_)
      for (const auto &fragment : batch.lanes)
        if (std::bit_cast<uint32_t>(fragment.linear_i) ||
            std::bit_cast<uint32_t>(fragment.linear_j))
          return false;
    return true;
  }

  static constexpr size_t fragment_lane_bytes() { return sizeof(GraphicsDraw::Fragment); }

  static void poison_retired_fragment_storage(GraphicsDraw &draw, size_t waves) {
    draw.fragments_.resize(waves);
    const float poison = std::bit_cast<float>(0x7f801234u);
    for (auto &batch : draw.fragments_)
      batch.lanes.fill({-17,
                        29,
                        poison,
                        poison,
                        poison,
                        poison,
                        poison,
                        {poison, poison, poison},
                        true,
                        0xab,
                        true});
    draw.fragments_.clear();
  }

  static std::array<size_t, 3> export_storage(const GraphicsDraw &draw) {
    return {draw.fragment_exports_.size(), draw.fragment_exports_.capacity(),
            reinterpret_cast<uintptr_t>(draw.fragment_exports_.data())};
  }

  static std::array<uint32_t, 5> exported(const GraphicsDraw &draw, uint32_t wave, uint32_t lane,
                                          uint32_t target) {
    const auto &value = draw.fragment_export(draw.fragments_.at(wave), lane, target);
    return {value.mask, value.values[0], value.values[1], value.values[2], value.values[3]};
  }

  static bool exports_prepared(const GraphicsDraw &draw) { return draw.fragment_exports_prepared_; }

  static bool covered(const GraphicsDraw &draw, uint32_t wave, uint32_t lane) {
    return draw.fragments_.at(wave).lanes[lane].covered;
  }

  static bool front(const GraphicsDraw &draw, uint32_t wave) {
    return draw.fragments_.at(wave).front;
  }

  static bool parallel_output(GraphicsDraw &draw, const GpuVmAccess &memory,
                              CpuDispatchPool &pool) {
    return draw.try_parallel_outputs(memory, pool, 4);
  }

  static void seed_early_depth_waves(GraphicsDraw &draw, uint32_t wave_size, bool greater) {
    draw.prepare_attachments();
    draw.fragment_stage_ = true;
    draw.fragment_wave_size_ = wave_size;
    draw.next_raster_group_ = draw.vertex_groups_.size();
    draw.fragments_.resize(3);
    for (uint32_t index = 0; index < 3; ++index) {
      auto &batch = draw.fragments_[index];
      batch.parameters = {0x100u + index};
      for (uint32_t lane = 0; lane < wave_size; ++lane) {
        auto &fragment = batch.lanes[lane];
        fragment.x = lane % 16;
        fragment.y = lane / 16;
        fragment.covered = lane % 4 != 3 && (index != 2 || lane < 17);
        fragment.z = (index == 1) != greater ? 0.25f : 0.75f;
        fragment.pull_model[2] = 1.0f;
      }
    }
  }

  static bool select_early_depth(GraphicsDraw &draw, const GpuVmAccess &memory) {
    return draw.select_fragment_waves(memory);
  }

  static std::vector<uint32_t> selected_fragments(const GraphicsDraw &draw) {
    return draw.fragment_dispatch_indices_;
  }

  static DispatchEntry selected_dispatch(GraphicsDraw &draw) {
    draw.prepare_fragment_exports();
    return draw.fragment_dispatch();
  }

  static void fragment_depth(GraphicsDraw &draw, uint32_t wave, uint32_t bits) {
    for (auto &fragment : draw.fragments_.at(wave).lanes)
      fragment.z = std::bit_cast<float>(bits);
  }

  static void early_depth_groups(GraphicsDraw &draw, uint32_t groups) {
    // Isolate fragment-window retirement from the separately tested VS ring
    // admission. No attributes are enabled in this scheduling witness.
    draw.vertex_group_limit_ = groups;
    draw.first_vertex_ = draw.instance_ = 0;
    draw.select_vertex_groups();
  }

  static void seed_fragment_inputs(GraphicsDraw &draw, uint32_t wave_size,
                                   bool zero_first_reciprocal = false) {
    draw.fragment_stage_ = true;
    draw.fragment_wave_size_ = wave_size;
    draw.fragments_.resize(1);
    auto &batch = draw.fragments_.front();
    batch = {};
    batch.relative_layer = 0x1234;
    batch.parameters = {0x01234567, 0x89abcdef, 0x7f801234};
    draw.sh_[0xb] = 2u << 1;
    draw.sh_[0xc] = 0x76543210;
    draw.sh_[0xd] = 0xfedcba98;
    for (uint32_t lane = 0; lane < 8; ++lane) {
      auto &f = batch.lanes[lane];
      f.x = int32_t(lane) - 3;
      f.y = 2 - int32_t(lane);
      f.i = std::bit_cast<float>(0x3f000001u + lane);
      f.j = std::bit_cast<float>(0x80000000u | lane);
      f.z = std::bit_cast<float>(0x7fc00000u + lane);
      f.linear_i = std::bit_cast<float>(1u + lane);
      f.linear_j = std::bit_cast<float>(0xbf800000u + lane);
      f.pull_model = {std::bit_cast<float>(0x7f800001u + lane), -0.0f,
                      zero_first_reciprocal && lane == 0 ? 0.0f : 1.0f};
      f.covered = (0xa5u >> lane) & 1;
    }
  }

  static void set_fragment_inputs(GraphicsDraw &draw, uint32_t mask) {
    draw.context_[0x198] = mask;
  }

  static bool gather_attributes(GraphicsDraw &draw, const GpuVmAccess &memory,
                                const std::array<uint32_t, 3> &indices, uint32_t provoking,
                                std::span<uint32_t> words, uint32_t group_offset = 0,
                                std::optional<uint64_t> ring_base = std::nullopt) {
    auto group = draw.vertex_groups_.front();
    group.attribute_offset = group_offset;
    if (ring_base)
      draw.attribute_ring_base_ = *ring_base;
    return draw.try_gather_attributes(memory, group, indices, provoking, words);
  }
};
} // namespace rocjitsu::amdgpu

namespace {

// Restore the environment already captured by each test without changing it on entry.
struct RestoreFenvAndErrno {
  std::fenv_t &environment;
  int error;
  ~RestoreFenvAndErrno() {
    std::fesetenv(&environment);
    errno = error;
  }
};

class ExportCollector final : public amdgpu::GraphicsStage {
public:
  struct Export {
    uint32_t lane, target, mask;
    std::array<uint32_t, 4> values;
  };
  std::vector<Export> exports;
  std::shared_ptr<amdgpu::GsRegisters> counters;
  std::shared_ptr<amdgpu::GsRegisters> gs_registers() const override { return counters; }
  void initialize(amdgpu::Wavefront &, uint32_t, uint32_t) override {}
  void export_mask(amdgpu::Wavefront &, uint64_t) override {}
  void export_lane(amdgpu::Wavefront &, uint32_t lane, uint32_t target, uint32_t mask,
                   const std::array<uint32_t, 4> &values) override {
    exports.push_back({lane, target, mask, values});
  }
};

class GraphicsBatchMemory final : public amdgpu::PhysicalMemoryAccess,
                                  public amdgpu::AddressSpaceTranslator {
public:
  mutable std::vector<std::byte> bytes = std::vector<std::byte>(0x400000);
  std::optional<std::pair<uint64_t, uint64_t>> alias_page;
  std::optional<uint64_t> missing_page;
  bool identity_available = true;
  uint64_t reads = 0, writes = 0;
  mutable uint64_t ram_lease_requests = 0;

  std::unique_ptr<amdgpu::VmRamLeaseRequest>
  prepare_ram_lease(amdgpu::PhysicalMemoryAccess &,
                    std::span<const amdgpu::VmRamRange>) const override {
    ++ram_lease_requests;
    return nullptr;
  }

  amdgpu::VmTranslationResult translate(uint64_t address, size_t size,
                                        amdgpu::VmAccessKind access) const override {
    if (missing_page && (address >> 12) == *missing_page)
      return {};
    return amdgpu::IdentityAddressSpaceTranslator{}.translate(address, size, access);
  }

  std::byte *pointer(uint64_t address, size_t size) const {
    if (missing_page && (address >> 12) == *missing_page)
      return nullptr;
    if (alias_page && (address >> 12) == alias_page->first)
      address = (alias_page->second << 12) | (address & 4095);
    return address <= bytes.size() && size <= bytes.size() - address ? bytes.data() + address
                                                                     : nullptr;
  }
  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain, uint64_t address,
                               std::span<std::byte> destination) override {
    ++reads;
    const auto *source = pointer(address, destination.size());
    if (!source)
      return amdgpu::VmAccessOutcome::Faulted;
    std::memcpy(destination.data(), source, destination.size());
    return amdgpu::VmAccessOutcome::Complete;
  }
  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain, uint64_t address,
                                std::span<const std::byte> source) override {
    ++writes;
    auto *destination = pointer(address, source.size());
    if (!destination)
      return amdgpu::VmAccessOutcome::Faulted;
    std::memcpy(destination, source.data(), source.size());
    return amdgpu::VmAccessOutcome::Complete;
  }
  std::byte *resolve_host_pointer(amdgpu::VmMemoryDomain, uint64_t address,
                                  size_t size) const override {
    return identity_available ? pointer(address, size) : nullptr;
  }
  std::pair<uint64_t, uint64_t> host_range(amdgpu::VmMemoryDomain,
                                           uint64_t address) const override {
    const auto *host = identity_available ? pointer(address & ~uint64_t{65535}, 65536) : nullptr;
    return host ? std::pair{reinterpret_cast<uint64_t>(host), uint64_t{65536}}
                : std::pair<uint64_t, uint64_t>{};
  }
};

class GraphicsExportTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  amdgpu::GpuMemory memory_{"graphics_memory"};
  amdgpu::GpuVm vm_;
  std::optional<amdgpu::GpuVmAccess> access_;
  amdgpu::L2Cache cache_{"graphics_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
  std::shared_ptr<ExportCollector> collector_ = std::make_shared<ExportCollector>();
  amdgpu::Lds lds_{4};

  void SetUp() override {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cache_.set_backing_memory(&memory_);
    cu_ = amdgpu::ComputeUnitCore::create("graphics_export", config, &memory_, &cache_);
    const auto address_space = vm_.register_address_space(
        0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
        std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_), {}, true);
    access_ = vm_.snapshot(address_space);
    ASSERT_TRUE(access_);
    cu_->set_gpu_vm(&vm_);
    decoder_ = Decoder::create(GetParam());
    wave_ = cu_->dispatch_wf(0, 0, 106, 16);
    wave_->set_graphics_stage(collector_);
    wave_->set_exec(3);
    wave_->set_lds(&lds_);
  }
  void TearDown() override { wave_->halt(); }

  template <size_t N> void run(const std::array<uint32_t, N> &words) {
    std::array<uint32_t, 4> padded{};
    std::copy(words.begin(), words.end(), padded.begin());
    auto decoded = decoder_->decode(padded.data());
    ASSERT_FALSE(decoded.failed());
    std::unique_ptr<Instruction> instruction(std::move(decoded).value());
    (void)cu_->execute_instruction(instruction.get(), *wave_);
  }

  void sample(uint8_t opcode, uint8_t dim, bool a16 = false, uint8_t mask = 15) {
    std::array<uint32_t, 4> words{};
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
      const auto encoded = rdna4::build_vsample(opcode, {.dim = dim,
                                                         .a16 = a16,
                                                         .dmask = mask,
                                                         .vdata = 12,
                                                         .rsrc = 8,
                                                         .samp = 4,
                                                         .vaddr0 = 0,
                                                         .vaddr1 = 1,
                                                         .vaddr2 = 2,
                                                         .vaddr3 = 3});
      std::copy(encoded.begin(), encoded.end(), words.begin());
    } else {
      const auto encoded = rdna3::build_mimg(
          opcode,
          {.dim = dim, .dmask = mask, .a16 = a16, .vaddr = 0, .vdata = 12, .srsrc = 2, .ssamp = 1});
      std::copy(encoded.begin(), encoded.end(), words.begin());
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(instruction->is_memory_op());
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    ASSERT_NE(instruction->data(), nullptr);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
  }

  // One rectangle instance, pixel-center inputs and full sample coverage.
  // Attachment, viewport and scissor state remain explicit in each test.
  amdgpu::Pm4QueueState rectangle_state() const {
    const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
    amdgpu::Pm4QueueState state;
    state.num_instances = 1;
    state.uconfig_registers[0x242] = 17;
    auto &ctx = state.context_registers;
    ctx[gfx12 ? 0x198 : 0x1b4] = 2;
    ctx[0x2f9] = 0x2d;
    ctx[gfx12 ? 0x205 : 0x206] = 0x43f;
    ctx[0x30e] = ctx[0x30f] = 0xffffffffu;
    return state;
  }

  amdgpu::Pm4QueueState batch_state(uint32_t wave_size = 32) const {
    auto state = rectangle_state();
    const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
    auto &ctx = state.context_registers;
    state.uconfig_registers[0x446] = 1; // 64 KiB ring at 0x10000.
    state.sh_registers[gfx12 ? 0x89 : 0xc8] = 0x2000;
    state.sh_registers[0x8] = 0x2100;
    state.sh_registers[0x8b] = (gfx12 ? 1u : 3u) << 16;
    ctx[gfx12 ? 0x2a6 : 0x2d5] = wave_size == 32 ? 1u << 22 : 0;
    ctx[gfx12 ? 0x190 : 0x1b6] = 1u << 15; // Wave32 fragment shader.
    ctx[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
    ctx[gfx12 ? 0x3b0 : 0x31c] = 10; // Linear RGBA8.
    ctx[0x318] = 0x1000;
    ctx[gfx12 ? 0x214 : 0x8e] = ctx[gfx12 ? 0x215 : 0x8f] = 15;
    ctx[gfx12 ? 0x195 : 0x1c5] = 9; // Full float exports.
    ctx[gfx12 ? 0x216 : 0x202] = 0xcc0010;
    ctx[gfx12 ? 0x198 : 0x1b4] = 0x302;  // Perspective center and position X/Y.
    ctx[gfx12 ? 0x1b : 0x203] = 1u << 4; // EARLY_Z_THEN_LATE_Z.
    ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(2.0f);
    ctx[0x91] = (4 - gfx12) | ((4 - gfx12) << 16);
    return state;
  }

  void batch_wave(uint32_t size, uint32_t group) {
    if (wave_->wf_size() != size) {
      wave_->halt();
      wave_ = cu_->dispatch_wf(0, 0, 106, 16, size);
    }
    wave_->set_lds(&lds_);
    wave_->set_wg_coord(group, 0, 0);
  }

  amdgpu::Pm4QueueState early_depth_state(uint32_t comparison = 1, bool writes = true) const {
    auto state = batch_state();
    auto &ctx = state.context_registers;
    const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
    ctx[gfx12 ? 5 : 7] = 15 | (3 << 16); // 16x4 D32.
    ctx[gfx12 ? 6 : 0x10] = 3;
    ctx[gfx12 ? 8 : 0x12] = ctx[gfx12 ? 10 : 0x14] = 0x1000;
    ctx[gfx12 ? 0x1c : 0x200] = 2 | (writes ? 4 : 0) | (comparison << 4);
    ctx[gfx12 ? 0x115 : 0xb4] = 0;
    ctx[gfx12 ? 0x116 : 0xb5] = 0x3f800000;
    ctx[0x318] = 0x1800;
    ctx[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (15 << 16) : 3 | (15 << 14);
    ctx[0x91] = (16 - gfx12) | ((4 - gfx12) << 16);
    return state;
  }

  void export_rectangle_vertices(amdgpu::GraphicsDraw &draw, uint32_t primitive = 0) {
    const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
    const uint32_t first = primitive * 3, index_bits = gfx12 ? 9 : 10;
    for (uint32_t i = 0; i < 3; ++i)
      draw.export_lane(*wave_, first + i, 12, 15,
                       {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                        std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                        std::bit_cast<uint32_t>(1.0f)});
    draw.export_lane(
        *wave_, primitive, 20, 1,
        {first | ((first + 1) << index_bits) | ((first + 2) << (2 * index_bits)), 0, 0, 0});
  }

  void initialize_fragment(const std::shared_ptr<amdgpu::GraphicsDraw> &draw) {
    wave_->set_wg_coord(0, 0, 0);
    wave_->set_graphics_stage(draw);
    draw->initialize(*wave_, 0, 0);
  }

  void execute(uint8_t target, uint8_t mask, bool row = false) {
    std::array<uint32_t, 2> words;
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
      words = rdna4::build_vexport({.en = mask,
                                    .tgt = target,
                                    .done = 1,
                                    .row_en = uint8_t(row),
                                    .vsrc0 = 3,
                                    .vsrc1 = 7,
                                    .vsrc2 = 9,
                                    .vsrc3 = 255});
    else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      words = rdna3_5::build_exp({.en = mask,
                                  .tgt = target,
                                  .done = 1,
                                  .row_en = uint8_t(row),
                                  .vsrc0 = 3,
                                  .vsrc1 = 7,
                                  .vsrc2 = 9,
                                  .vsrc3 = 255});
    else
      words = rdna3::build_exp({.en = mask,
                                .tgt = target,
                                .done = 1,
                                .row_en = uint8_t(row),
                                .vsrc0 = 3,
                                .vsrc1 = 7,
                                .vsrc2 = 9,
                                .vsrc3 = 255});
    run(words);
  }
};

// Sealed private storage is an admission requirement, not a generic translator
// promise. Keep this backing alive across the late ordered attachment visits.
struct EarlyDepthRam {
  static constexpr uint64_t base = 0x100000;
  std::vector<uint8_t> bytes = std::vector<uint8_t>(1 << 20);
  amdgpu::LegacyPageTable table;
  util::DistributedSharedMutex table_mutex;
  std::shared_ptr<util::DistributedSharedMutex> request_mutex =
      std::make_shared<util::DistributedSharedMutex>();
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter;
  std::optional<amdgpu::GpuVmAccess> access;
  explicit EarlyDepthRam(amdgpu::GpuMemory &memory) : adapter(vm, &memory) {
    for (size_t offset = 0; offset < bytes.size(); offset += 4096)
      table[(base + offset) >> 12] = {bytes.data() + offset, amdgpu::Mtype::RW,
                                      amdgpu::LegacyHostExtentOwner::DriverSealedRam};
    access = vm.snapshot(
        adapter.register_address_space(7, &table, &table_mutex, nullptr, request_mutex));
    fill(0x3f000000);
  }
  void fill(uint32_t depth) {
    for (size_t offset = 0; offset < bytes.size(); offset += sizeof(depth))
      std::memcpy(bytes.data() + offset, &depth, sizeof(depth));
  }
};

TEST_P(GraphicsExportTest, EarlyDepthKeepsWholeWaveInputsAndOrderedOutputs) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  EarlyDepthRam ram(memory_);
  ASSERT_TRUE(ram.access);
  for (uint32_t wave_size : {32u, 64u}) {
    for (uint32_t comparison : {1u, 3u, 4u, 6u}) {
      for (bool writes : {false, true}) {
        SCOPED_TRACE(testing::Message() << wave_size << ',' << comparison << ',' << writes);
        ram.fill(0x3f000000);
        const auto initial = ram.bytes;
        auto original = std::make_shared<amdgpu::GraphicsDraw>(
            early_depth_state(comparison, writes), GetParam(), 3);
        Access::seed_early_depth_waves(*original, wave_size, comparison >= 4);
        auto selected = std::make_shared<amdgpu::GraphicsDraw>(*original);
        const auto inputs = Access::fragment_bits(*original);
        ASSERT_TRUE(Access::select_early_depth(*selected, *ram.access));
        EXPECT_EQ(Access::selected_fragments(*selected), (std::vector<uint32_t>{1}));
        EXPECT_EQ(Access::selected_dispatch(*selected).total_wgs, 1u);
        EXPECT_EQ(Access::selected_dispatch(*original).total_wgs, 3u);
        EXPECT_EQ(Access::export_storage(*selected)[0], 64u);
        EXPECT_EQ(Access::export_storage(*original)[0], 3u * 64u);
        EXPECT_EQ(Access::fragment_bits(*selected), inputs);
        EXPECT_EQ(ram.bytes, initial);
        // The surviving wave keeps the original lane/helper configuration.
        batch_wave(wave_size, 1);
        original->initialize(*wave_, 1, 0);
        const uint64_t expected_exec = wave_->exec();
        batch_wave(wave_size, 0);
        selected->initialize(*wave_, 0, 0);
        EXPECT_EQ(wave_->exec(), expected_exec);
        for (uint32_t index = 0; index < 3; ++index) {
          batch_wave(wave_size, index);
          for (uint32_t lane = 0; lane < wave_size; ++lane)
            original->export_lane(*wave_, lane, 0, 15, {0x3f800000, 0x3e800000, 0, 0x3f800000});
        }
        batch_wave(wave_size, 0);
        for (uint32_t lane = 0; lane < wave_size; ++lane)
          selected->export_lane(*wave_, lane, 0, 15, {0x3f800000, 0x3e800000, 0, 0x3f800000});
        EXPECT_FALSE(original->advance(*ram.access));
        const auto expected = ram.bytes;
        EXPECT_NE(expected, initial);
        std::copy(initial.begin(), initial.end(), ram.bytes.begin());
        EXPECT_FALSE(selected->advance(*ram.access));
        EXPECT_EQ(ram.bytes, expected);
      }
    }
  }
}

TEST_P(GraphicsExportTest, CompactFragmentExportsPreserveMasksAndResetAfterRetirement) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  EXPECT_EQ(Access::fragment_lane_bytes(), 44u);
  EarlyDepthRam ram(memory_);
  ASSERT_TRUE(ram.access);
  for (uint32_t wave_size : {32u, 64u}) {
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(early_depth_state(0), GetParam(), 3);
    std::array<size_t, 3> previous{};
    for (uint32_t window = 0; window < 2; ++window) {
      Access::seed_early_depth_waves(*draw, wave_size, false);
      ASSERT_EQ(Access::selected_dispatch(*draw).total_wgs, 3u);
      const auto storage = Access::export_storage(*draw);
      ASSERT_EQ(storage[0], 3u * 64u);
      if (window) {
        EXPECT_EQ(storage, previous); // Reuse capacity, but not the previous exports.
      }
      for (uint32_t group = 0; group < 3; ++group) {
        batch_wave(wave_size, group);
        for (uint32_t lane = 0; lane < 64; ++lane)
          EXPECT_EQ(Access::exported(*draw, group, lane, 0), (std::array<uint32_t, 5>{}));
        for (uint32_t lane = 0; lane < wave_size; ++lane) {
          const uint32_t value = 0x3f000000 + group * 256 + lane + window;
          draw->export_lane(*wave_, lane, 0, 1, {value, 2, 3, 4});
          draw->export_lane(*wave_, lane, 0, 4, {5, 6, value + 1, 8});
          draw->export_lane(*wave_, lane, 0, 1, {value + 2, 10, 11, 12});
          draw->export_lane(*wave_, lane, 0, 0, {13, 14, 15, 16});
          draw->export_lane(*wave_, lane, 7, 15, {17, 18, 19, 20});
          EXPECT_EQ(Access::exported(*draw, group, lane, 0),
                    (std::array<uint32_t, 5>{5, value + 2, 0, value + 1, 0}));
          EXPECT_EQ(Access::exported(*draw, group, lane, 7), (std::array<uint32_t, 5>{}));
        }
      }
      EXPECT_EQ(Access::export_storage(*draw), storage);
      const auto depth = ram.bytes;
      // NEVER rejects at the original late depth visit, before reading exports.
      EXPECT_FALSE(draw->advance(*ram.access));
      EXPECT_EQ(ram.bytes, depth);
      EXPECT_EQ(Access::export_storage(*draw)[0], 0u);
      previous = storage;
    }
  }
}

TEST_P(GraphicsExportTest, CompactFragmentExportsRetainUnboundOperandAndCoverageChecks) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  auto draw = std::make_shared<amdgpu::GraphicsDraw>(early_depth_state(), GetParam(), 3);
  Access::seed_early_depth_waves(*draw, 32, false);
  ASSERT_EQ(Access::selected_dispatch(*draw).total_wgs, 3u);
  batch_wave(32, 0);
  wave_->set_graphics_stage(draw);
  wave_->set_exec(1);
  wave_->debug_write_vgpr(3, 0, 0x3f800000);
  const auto storage = Access::export_storage(*draw);
  wave_->export_graphics(7, 1, {3, 0, 0, 0}, false);
  EXPECT_FALSE(wave_->instruction_execution_failed());
  EXPECT_TRUE(Access::covered(*draw, 0, 0));
  EXPECT_FALSE(Access::covered(*draw, 0, 1));
  EXPECT_EQ(Access::exported(*draw, 0, 0, 7), (std::array<uint32_t, 5>{}));
  wave_->export_graphics(7, 1, {wave_->num_vgprs(), 0, 0, 0}, false);
  EXPECT_TRUE(wave_->instruction_execution_failed());
  wave_->clear_instruction_execution_error();
  wave_->export_graphics(9, 1, {3, 0, 0, 0}, false);
  EXPECT_TRUE(wave_->instruction_execution_failed());
  EXPECT_EQ(Access::export_storage(*draw), storage);
}

TEST_P(GraphicsExportTest, CompactFragmentExportsDepthOnlyNeedsNoColorStorage) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  EarlyDepthRam ram(memory_);
  ASSERT_TRUE(ram.access);
  for (uint32_t wave_size : {32u, 64u}) {
    ram.fill(0x3f000000);
    auto state = early_depth_state();
    state.context_registers[GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? 0x216 : 0x202] = 0;
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    Access::seed_early_depth_waves(*draw, wave_size, false);
    ASSERT_EQ(Access::selected_dispatch(*draw).total_wgs, 3u);
    EXPECT_TRUE(Access::exports_prepared(*draw));
    EXPECT_EQ(Access::export_storage(*draw)[0], 0u);
    for (uint32_t group = 0; group < 3; ++group) {
      batch_wave(wave_size, group);
      for (uint32_t lane = 0; lane < wave_size; ++lane) {
        draw->export_lane(*wave_, lane, 0, 15, {1, 2, 3, 4});
        draw->export_lane(*wave_, lane, 8, 1, {0x3e000000, 0, 0, 0});
        draw->export_lane(*wave_, lane, 8, 0, {0x3f800000, 0, 0, 0});
        EXPECT_EQ(Access::exported(*draw, group, lane, 0), (std::array<uint32_t, 5>{}));
      }
    }
    draw->export_lane(*wave_, 0, 8, 2, {});
    EXPECT_TRUE(wave_->instruction_execution_failed());
    wave_->clear_instruction_execution_error();
    EXPECT_FALSE(draw->advance(*ram.access));
    uint32_t depth = 0;
    std::memcpy(&depth, ram.bytes.data(), sizeof(depth));
    EXPECT_EQ(depth, 0x3e000000u);
    EXPECT_FALSE(Access::exports_prepared(*draw));
    EXPECT_EQ(Access::export_storage(*draw)[0], 0u);
  }
}

TEST_P(GraphicsExportTest, CompactFragmentExportsGiveConcurrentWavesDisjointStorage) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  auto draw = std::make_shared<amdgpu::GraphicsDraw>(early_depth_state(), GetParam(), 3);
  Access::seed_early_depth_waves(*draw, 64, false);
  ASSERT_EQ(Access::selected_dispatch(*draw).total_wgs, 3u);
  const auto storage = Access::export_storage(*draw);
  std::array<std::unique_ptr<amdgpu::ComputeUnitCore>, 3> owners;
  std::array<amdgpu::Wavefront *, 3> waves{};
  for (uint32_t group = 0; group < owners.size(); ++group) {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam();
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    owners[group] = amdgpu::ComputeUnitCore::create("export_owner_" + std::to_string(group), config,
                                                    &memory_, &cache_);
    waves[group] = owners[group]->dispatch_wf(group, 0, 106, 16, 64);
    ASSERT_NE(waves[group], nullptr);
    waves[group]->set_wg_coord(group, 0, 0);
  }
  amdgpu::CpuDispatchPool pool(3);
  pool.run_indexed(3, 3, [&](size_t group) {
    for (uint32_t repeat = 0; repeat < 64; ++repeat)
      for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t value = (uint32_t(group) << 24) | (repeat << 8) | lane;
        draw->export_lane(*waves[group], lane, 0, 3, {value, ~value, 0, 0});
        draw->export_lane(*waves[group], lane, 0, 4, {0, 0, value + 1, 0});
      }
  });
  EXPECT_EQ(Access::export_storage(*draw), storage);
  for (uint32_t group = 0; group < waves.size(); ++group) {
    for (uint32_t lane = 0; lane < 64; ++lane) {
      const uint32_t value = (group << 24) | (63u << 8) | lane;
      EXPECT_EQ(Access::exported(*draw, group, lane, 0),
                (std::array<uint32_t, 5>{7, value, ~value, value + 1, 0}));
    }
    waves[group]->halt();
  }
}

TEST_P(GraphicsExportTest, EarlyDepthRetiresRejectedWindowBeforeNextRasterGroup) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  EarlyDepthRam ram(memory_);
  ASSERT_TRUE(ram.access);
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto state = early_depth_state();
  auto &ctx = state.context_registers;
  ctx[gfx12 ? 0x216 : 0x202] = 0; // Depth only; no shader export is required.
  ctx[gfx12 ? 5 : 7] = 63 | (63 << 16);
  ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] = 0x42000000; // 32.
  ctx[0x113] = 0x3f800000;
  ctx[0x91] = (64 - gfx12) | ((64 - gfx12) << 16);
  amdgpu::GraphicsDraw draw(state, GetParam(), 150);
  Access::early_depth_groups(draw, 5);
  ASSERT_EQ(draw.vertex_dispatch().total_wgs, 5u);
  for (uint32_t group = 0; group < 5; ++group) {
    batch_wave(32, group);
    draw.initialize(*wave_, group, 0);
    for (uint32_t lane = 0; lane < 30; ++lane)
      draw.export_lane(*wave_, lane, 12, 15,
                       {std::bit_cast<uint32_t>(lane % 3 == 2 ? 1.0f : -1.0f),
                        std::bit_cast<uint32_t>(lane % 3 == 1 ? 1.0f : -1.0f),
                        group == 4 ? 0x3e800000u : 0x3f400000u, 0x3f800000});
    for (uint32_t lane = 0; lane < 10; ++lane)
      draw.export_lane(*wave_, lane, 20, 1, {wave_->debug_read_vgpr(0, lane), 0, 0, 0});
  }
  const auto initial = ram.bytes;
  // Four groups exceed the 4096-wave window and all fail. The fifth group
  // must still be rasterized and published, without an empty FS dispatch.
  const auto fragment = draw.advance(*ram.access, nullptr, 1, false, true);
  ASSERT_TRUE(fragment);
  EXPECT_TRUE(draw.fragment_stage());
  EXPECT_EQ(fragment->total_wgs, 10u * 64u * 64u / 32u);
  EXPECT_EQ(ram.bytes, initial);
  EXPECT_FALSE(draw.advance(*ram.access, nullptr, 1, false, true));
  uint32_t final_depth = 0;
  std::memcpy(&final_depth, ram.bytes.data(), sizeof(final_depth));
  EXPECT_EQ(final_depth, 0x3e800000u);
  EXPECT_NE(ram.bytes, initial);
}

TEST_P(GraphicsExportTest, EarlyDepthStateAndUnknownBackingDeclineBeforeReads) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  auto recording = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  const auto handle = vm.register_address_space(7, recording, recording, {}, true);
  auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto check_state = [&](amdgpu::Pm4QueueState state) {
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    Access::seed_early_depth_waves(*draw, 32, false);
    const auto before = Access::fragment_bits(*draw);
    const auto requests = recording->ram_lease_requests;
    EXPECT_FALSE(Access::select_early_depth(*draw, *access));
    EXPECT_EQ(recording->ram_lease_requests, requests);
    EXPECT_EQ(Access::fragment_bits(*draw), before);
    EXPECT_EQ(Access::selected_dispatch(*draw).total_wgs, 3u);
  };
  for (uint32_t bit : {0u, 1u, 2u, 5u, 6u, 7u, 8u, 9u, 10u, 12u, 13u, 16u, 23u}) {
    auto state = early_depth_state();
    state.context_registers[gfx12 ? 0x1b : 0x203] |= 1u << bit;
    check_state(std::move(state));
  }
  for (uint32_t comparison : {0u, 2u, 5u, 7u})
    check_state(early_depth_state(comparison));
  for (uint32_t mode = 0; mode < 11; ++mode) {
    auto state = early_depth_state();
    if (mode == 0)
      state.context_registers[gfx12 ? 0x315 : 0x313] = 1u;
    else if (mode == 1)
      state.performance_counters_active = true;
    else if (mode == 2)
      state.unsupported_pixel_counter_mode = true;
    else if (mode == 3)
      state.context_registers[gfx12 ? 0x18 : 1] = 0x11000106;
    else if (mode == 4)
      state.context_registers[0x2f8] = 1u << 20;
    else if (mode == 5)
      state.context_registers[3] = 1u << 6;
    else if (mode == 6)
      state.context_registers[gfx12 ? 0x1e : 0x201] = 1u << 8;
    else if (mode == 7)
      state.context_registers[0x293] = 1u << 16;
    else if (mode == 8)
      state.context_registers[gfx12 ? 0x194 : 0x1c4] = 1;
    else if (mode == 9)
      state.context_registers[0] = 1u << 16;
    else
      state.context_registers[3] = 1u << 30;
    check_state(std::move(state));
  }
  for (uint32_t bit : {5u, 21u, 22u, 23u}) {
    auto state = early_depth_state();
    state.context_registers[gfx12 ? 0x315 : 0x313] = (1u << 20) | (1u << bit);
    check_state(std::move(state));
  }
  uint32_t requests = 0;
  for (uint32_t conservative : {0u, 1u << 20}) {
    auto state = early_depth_state();
    state.context_registers[gfx12 ? 0x315 : 0x313] = conservative;
    state.context_registers[3] = gfx12 ? 1u << 12 : 1u << 16;
    state.context_registers[0] = 0x00f00060u;
    auto ordinary = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    Access::seed_early_depth_waves(*ordinary, 32, false);
    EXPECT_FALSE(Access::select_early_depth(*ordinary, *access));
    EXPECT_EQ(recording->ram_lease_requests, ++requests);
  }
  EXPECT_EQ(recording->reads, 0u);
  EXPECT_EQ(recording->writes, 0u);
}

TEST_P(GraphicsExportTest, EarlyDepthStrictBackingAndLateFaultsStayOrdered) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  EarlyDepthRam ram(memory_);
  ASSERT_TRUE(ram.access);
  auto make_draw = [&] {
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(early_depth_state(), GetParam(), 3);
    Access::seed_early_depth_waves(*draw, 32, false);
    return draw;
  };
  const auto initial = ram.bytes;
  const uint64_t page = EarlyDepthRam::base >> 12;
  const auto entry = ram.table.at(page);
  auto *host = entry.host_extents.front().host_ptr;
  constexpr auto sealed = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
  for (uint32_t variant = 0; variant < 5; ++variant) {
    ram.table[page] = entry;
    ram.table[0x180] = {ram.bytes.data() + 0x80000, amdgpu::Mtype::RW, sealed};
    if (variant == 0)
      ram.table[page].host_extents.front().owner = amdgpu::LegacyHostExtentOwner::Application;
    else if (variant == 1)
      ram.table[page].host_extents = {{host, 2, 0, sealed}, {host + 2, 4094, 2, sealed}};
    else if (variant == 2)
      ram.table[page].host_extents.push_back({host + 8, 8, 8, sealed});
    else if (variant == 3)
      ram.table.erase(page);
    else
      ram.table[0x180].host_extents.front().host_ptr = host;
    auto draw = make_draw();
    EXPECT_FALSE(Access::select_early_depth(*draw, *ram.access));
    EXPECT_EQ(ram.bytes, initial);
  }
  ram.table[page] = entry;
  ram.table[0x180] = {ram.bytes.data() + 0x80000, amdgpu::Mtype::RW, sealed};
  auto selected = make_draw();
  // All waves may be omitted, but the original first depth read is retained.
  Access::fragment_depth(*selected, 1, 0x3f400000);
  ASSERT_TRUE(Access::select_early_depth(*selected, *ram.access));
  EXPECT_TRUE(Access::selected_fragments(*selected).empty());
  EXPECT_FALSE(Access::exports_prepared(*selected));
  EXPECT_EQ(Access::export_storage(*selected)[0], 0u);
  ram.table.erase(page);
  EXPECT_THROW(selected->advance(*ram.access), std::runtime_error);
  EXPECT_FALSE(Access::exports_prepared(*selected));
  EXPECT_EQ(Access::export_storage(*selected)[0], 0u);
  EXPECT_EQ(ram.bytes, initial);
}

TEST_P(GraphicsExportTest, EarlyDepthIntegerPredicatePreservesHostStateAndBoundaryFallback) {
  using Access = amdgpu::GraphicsDrawTestAccess;
  EarlyDepthRam ram(memory_);
  ASSERT_TRUE(ram.access);
  struct Case {
    uint32_t incoming, previous, comparison;
    bool omitted;
  };
  constexpr Case cases[] = {{0, 0x80000000, 1, true},          {0x80000000, 0, 3, false},
                            {0x40000000, 0x3f000000, 1, true}, // Clamp to one.
                            {0xbf800000, 0x3f000000, 4, true}, // Clamp to zero.
                            {1, 0x3f000000, 1, false},         {0x7f800001, 0x3f000000, 1, false},
                            {0x3f400000, 1, 1, false},         {0x3f400000, 0x7fc00000, 1, false}};
  std::fenv_t original;
  std::fegetenv(&original);
  const int original_errno = errno;
  RestoreFenvAndErrno restore{original, original_errno};
  for (const auto &entry : cases) {
    for (bool traps : {false, true}) {
#if !defined(__GLIBC__) || !defined(__x86_64__)
      if (traps)
        continue;
#endif
      std::fesetenv(FE_DFL_ENV);
      ram.fill(entry.previous);
      auto draw = std::make_shared<amdgpu::GraphicsDraw>(early_depth_state(entry.comparison, false),
                                                         GetParam(), 3);
      Access::seed_early_depth_waves(*draw, 64, false);
      for (uint32_t wave = 0; wave < 3; ++wave)
        Access::fragment_depth(*draw, wave, entry.incoming);
      std::fesetround(FE_DOWNWARD);
      std::feraiseexcept(FE_INEXACT);
#if defined(__GLIBC__) && defined(__x86_64__)
      if (traps)
        feenableexcept(FE_INVALID | FE_DIVBYZERO);
#endif
      std::fenv_t before, after;
      std::fegetenv(&before);
      errno = EDOM;
      const bool omitted = Access::select_early_depth(*draw, *ram.access);
      const int observed_errno = errno;
      std::fegetenv(&after);
      const int rounding = std::fegetround();
      const int flags = std::fetestexcept(FE_ALL_EXCEPT);
      // Reset enabled traps before test-framework formatting or the next fixture.
      std::fesetenv(FE_DFL_ENV);
      EXPECT_EQ(omitted, entry.omitted);
      EXPECT_EQ(observed_errno, EDOM);
      EXPECT_EQ(rounding, FE_DOWNWARD);
      EXPECT_EQ(flags, FE_INEXACT);
#if defined(__GLIBC__) && defined(__x86_64__)
      EXPECT_EQ(after.__control_word, before.__control_word);
      EXPECT_EQ(after.__status_word, before.__status_word);
      EXPECT_EQ(after.__mxcsr, before.__mxcsr);
#endif
    }
  }
}

TEST_P(GraphicsExportTest, GraphicsRegisterAccessHonorsPendingLanesAndPackedBytes) {
  auto config = cu_->config();
  config.memory_wait_diagnostics = amdgpu::MemoryWaitDiagnostics::Warn;
  auto cu = amdgpu::ComputeUnitCore::create("graphics_wait", config, &memory_, &cache_);
  auto *wave = cu->dispatch_wf(0, 0, 106, 16);
  ASSERT_NE(wave, nullptr);
  wave->set_graphics_stage(collector_);
  wave->set_exec(1);
  wave->set_lds(&lds_);
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  // Linear 1D image: A16 consumes only the coordinate's low half.
  const std::array<uint32_t, 8> descriptor{
      0x1000, (20u << (gfx12 ? 17 : 20)) | (1u << 30), 0, (8u << 28) | 0xfac, 0, 0, 0, 0};
  for (uint32_t r = 0; r < descriptor.size(); ++r)
    wave->debug_write_sgpr(8 + r, descriptor[r]);
  wave->debug_write_vgpr(0, 0, 0xfeed0000);
  enum class Operation { Interpolate, Export, Parameter, Image };
  struct Case {
    Operation operation;
    uint16_t reg;
    uint32_t lane;
    uint8_t bytes = 0xf;
    bool expected = true;
  };
  const Case cases[] = {{Operation::Interpolate, 0, 1}, // P10 comes from an inactive lane.
                        {Operation::Interpolate, 0, 3, 0xf, false},
                        {Operation::Interpolate, 8, 0}, // Destination write conflicts too.
                        {Operation::Export, 3, 0},
                        {Operation::Export, 3, 1, 0xf, false},
                        {Operation::Export, 7, 0, 0xf, false}, // Masked component.
                        {Operation::Parameter, 8, 2}, // Active quad expands into three lanes.
                        {Operation::Parameter, 8, 3, 0xf, false},
                        {Operation::Image, 0, 0, 0x3},
                        {Operation::Image, 0, 0, 0xc, false},
                        {Operation::Image, 0, 1, 0x3, false}};
  auto &state = wave->ensure_memory_wait_scoreboard();
  for (const auto &test : cases) {
    for (bool waited : {false, true}) {
      SCOPED_TRACE(testing::Message() << int(test.operation) << ',' << test.reg << ',' << test.lane
                                      << ',' << int(test.bytes) << ',' << waited);
      state.clear();
      state.add({state.issue(WaitCounterKind::Load),
                 0x100,
                 uint64_t{1} << test.lane,
                 {RegClass::VGPR, test.reg, 1},
                 WaitCounterKind::Load,
                 test.bytes});
      unsigned reports = 0;
      state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      if (waited)
        state.wait(WaitCounterKind::Load, 0);
      const amdgpu::ScopedMemoryWaitCheck check(&state);
      switch (test.operation) {
      case Operation::Interpolate:
        amdgpu::execute_graphics_interp(*wave, 8, {0, 1, 2}, false, 0, false, 0);
        break;
      case Operation::Export:
        wave->export_graphics(0, 1, {3, 7, 9, 15}, false);
        break;
      case Operation::Parameter:
        amdgpu::execute_graphics_parameter_load(*wave, 8, 0, 0);
        break;
      case Operation::Image: {
        amdgpu::VectorMemState data(amdgpu::GLOBAL_MEM);
        data.is_load = true;
        ASSERT_TRUE(amdgpu::prepare_image_transfer(*wave, data, 8, 12, {0, 1, 2}, 0, 1, false,
                                                   false, ~0u, amdgpu::ImageSampleMode::Implicit,
                                                   true));
        break;
      }
      }
      EXPECT_EQ(reports != 0, test.expected && !waited);
    }
  }
  wave->halt();
}

TEST_P(GraphicsExportTest, GsCounterInstructionsAccumulateAtBothWidths) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 maintains streamout counters in ordinary memory";
  collector_->counters = std::make_shared<amdgpu::GsRegisters>();
  amdgpu::LocalMemPipeline pipeline;
  for (uint32_t index = 0; index < 16; ++index) {
    uint64_t value = 0;
    for (const auto &[subtract, operand] :
         {std::pair{false, UINT32_MAX}, std::pair{false, 3u}, std::pair{true, 9u}}) {
      SCOPED_TRACE(index);
      wave_->set_exec(0x38); // Lane three supplies the operand, even with other active lanes.
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        wave_->debug_write_vgpr(0, lane, lane == 3 ? operand : 17);
        wave_->debug_write_vgpr(2, lane, 0xdeadbeef);
        wave_->debug_write_vgpr(3, lane, 0x12345678);
      }
      const uint8_t opcode = subtract ? 123 : 122;
      const auto words =
          GetParam() == ROCJITSU_CODE_ARCH_RDNA3
              ? rdna3::build_ds(opcode,
                                {.offset0 = uint8_t(index * 4), .gds = 1, .data0 = 0, .vdst = 2})
              : rdna3_5::build_ds(opcode,
                                  {.offset0 = uint8_t(index * 4), .gds = 1, .data0 = 0, .vdst = 2});
      auto decoded = decoder_->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(instruction->is_memory_op());
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      ASSERT_FALSE(wave_->instruction_execution_failed());
      EXPECT_EQ(pipeline.issue(instruction.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
      EXPECT_TRUE(wave_->wait_counters().empty());
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        EXPECT_EQ(wave_->debug_read_vgpr(2, lane), lane == 3 ? uint32_t(value) : 0xdeadbeef);
        EXPECT_EQ(wave_->debug_read_vgpr(3, lane),
                  lane == 3 && index >= 8 ? uint32_t(value >> 32) : 0x12345678);
      }
      value = subtract ? value - operand : value + operand;
      if (index < 8)
        value = uint32_t(value);
      EXPECT_EQ(collector_->counters->modify(index, 0, false), value);
    }
  }
}

TEST_P(GraphicsExportTest, GsCounterEmptyExecPreservesRegistersAndCounters) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 maintains streamout counters in ordinary memory";
  collector_->counters = std::make_shared<amdgpu::GsRegisters>();
  collector_->counters->modify(8, 42, false);
  wave_->set_exec(0);
  wave_->debug_write_vgpr(0, 0, 99);
  wave_->debug_write_vgpr(2, 0, 0xdeadbeef);
  const auto words = rdna3::build_ds(122, {.offset0 = 32, .gds = 1, .data0 = 0, .vdst = 2});
  auto decoded = decoder_->decode(words.data());
  ASSERT_FALSE(decoded.failed());
  auto instruction = std::move(decoded).value();
  ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
  ASSERT_FALSE(wave_->instruction_execution_failed());
  amdgpu::LocalMemPipeline pipeline;
  EXPECT_EQ(pipeline.issue(instruction.release(), *wave_), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(collector_->counters->streamout_stats(0)[0], 42u);
  EXPECT_EQ(wave_->debug_read_vgpr(2, 0), 0xdeadbeef);
}

TEST_P(GraphicsExportTest, GsCounterRequiresStageStateAndValidRegisterRanges) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 maintains streamout counters in ordinary memory";
  for (uint32_t variant = 0; variant < 4; ++variant) {
    SCOPED_TRACE(variant);
    wave_->clear_instruction_execution_error();
    collector_->counters = variant ? std::make_shared<amdgpu::GsRegisters>() : nullptr;
    const auto words = rdna3::build_ds(122, {.offset0 = 32,
                                             .gds = uint8_t(variant != 3),
                                             .data0 = uint8_t(variant == 1 ? 16 : 0),
                                             .vdst = uint8_t(variant == 2 ? 15 : 2)});
    if (variant == 3) {
      EXPECT_THROW(run(words), util::UnimplementedInst);
    } else {
      run(words);
      EXPECT_TRUE(wave_->instruction_execution_failed());
    }
    if (collector_->counters) {
      EXPECT_EQ(collector_->counters->streamout_stats(0)[0], 0u);
    }
    EXPECT_TRUE(wave_->wait_counters().empty());
  }
}

TEST_P(GraphicsExportTest, HalfInterpolationResultsHonorRoundingMode) {
  constexpr uint16_t positive[] = {0x3c01, 0x3c01, 0x3c00, 0x3c00};
  constexpr uint16_t negative[] = {0xbc01, 0xbc00, 0xbc01, 0xbc00};
  wave_->set_exec(0x7fffffff);
  for (uint32_t mode = 0; mode < 4; ++mode) {
    wave_->set_mode_raw(mode << 2);
    for (bool high : {false, true}) {
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        // Halfway between the half-precision midpoint and its upper neighbor.
        wave_->debug_write_vgpr(3, lane, 0x3f801800u | ((lane & 1) << 31));
        wave_->debug_write_vgpr(6, lane, 0x12345678);
      }
      const uint8_t opcode = high ? 34 : 33;
      if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
        run(rdna4::build_vop3p(opcode, {.vdst = 6, .src0 = 259, .src1 = 242, .src2 = 128}));
      else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
        run(rdna3_5::build_vop3p(opcode, {.vdst = 6, .src0 = 259, .src1 = 242, .src2 = 128}));
      else
        run(rdna3::build_vop3p(opcode, {.vdst = 6, .src0 = 259, .src1 = 242, .src2 = 128}));
      for (uint32_t lane = 0; lane < 31; ++lane) {
        const uint32_t half = lane & 1 ? negative[mode] : positive[mode];
        EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 6, lane),
                  high ? (half << 16) | 0x5678 : 0x12340000 | half)
            << "mode=" << mode << " high=" << high << " lane=" << lane;
      }
      EXPECT_EQ(cu_->read_vgpr(wave_->vgpr_alloc().base + 6, 31), 0x12345678u);
    }
  }
}

TEST_P(GraphicsExportTest, ParallelColorTilesPreserveOverlapAndHostEnvironment) {
#if defined(__GLIBC__) && defined(__x86_64__)
  constexpr uint32_t width = 128, height = 64;
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  amdgpu::LegacyPageTable table;
  util::DistributedSharedMutex table_mutex;
  auto request_mutex = std::make_shared<util::DistributedSharedMutex>();
  std::vector<uint8_t> storage(65536);
  for (size_t page = 0; page < storage.size() / 4096; ++page)
    table[0x100 + page] = {storage.data() + page * 4096, amdgpu::Mtype::RW,
                           amdgpu::LegacyHostExtentOwner::DriverSealedRam};
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter(vm, &memory_);
  const auto handle =
      adapter.register_address_space(7, &table, &table_mutex, nullptr, request_mutex);
  auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  amdgpu::CpuDispatchPool pool(4);
  std::fenv_t saved_environment;
  std::fegetenv(&saved_environment);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved_environment, saved_errno};
  for (uint32_t blend : {0u, (1u << 30) | 1u | (1u << 8)}) {
    for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      SCOPED_TRACE(blend);
      SCOPED_TRACE(rounding);
      auto state = rectangle_state();
      auto &ctx = state.context_registers;
      ctx[0x318] = 0x1000;
      ctx[gfx12 ? 0x3b0 : 0x31c] = 10;
      ctx[gfx12 ? 0x31e : 0x3b0] =
          gfx12 ? ((width - 1) << 16) | (height - 1) : ((width - 1) << 14) | (height - 1);
      ctx[gfx12 ? 0x31f : 0x3b8] = gfx12 ? 3 << 15 : 26 << 14;
      ctx[gfx12 ? 0x214 : 0x8e] = ctx[gfx12 ? 0x215 : 0x8f] = 15;
      ctx[gfx12 ? 0x195 : 0x1c5] = 4;
      ctx[gfx12 ? 0x216 : 0x202] = 0xcc0010;
      ctx[0x1e0] = blend;
      ctx[0x10f] = ctx[0x110] = std::bit_cast<uint32_t>(float(width / 2));
      ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(float(height / 2));
      ctx[0x91] = (width - gfx12) | ((height - gfx12) << 16);
      auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 6);
      for (uint32_t primitive = 0; primitive < 2; ++primitive)
        export_rectangle_vertices(*draw, primitive);
      const auto dispatch = draw->advance(*access);
      ASSERT_TRUE(dispatch);
      ASSERT_GE(dispatch->total_wgs, 128u);
      for (uint32_t workgroup = 0; workgroup < dispatch->total_wgs; ++workgroup) {
        wave_->set_wg_coord(workgroup, 0, 0);
        wave_->set_graphics_stage(draw);
        draw->initialize(*wave_, workgroup, 0);
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
          draw->export_lane(
              *wave_, lane, 0, 3,
              {workgroup < dispatch->total_wgs / 2 ? 0x34003000u : 0x38003400u, 0x3c003400u, 0, 0});
      }
      std::fill(storage.begin(), storage.end(), 0);
      const auto initial = storage;
      if (!blend && rounding == FE_TONEAREST) {
        // A physically contiguous split inside a later pixel is still a strict
        // transfer boundary. Optional tiling must leave the serial prefix intact.
        const uint64_t split_page = 0x102;
        const auto original_entry = table.at(split_page);
        auto *host = original_entry.host_extents.front().host_ptr;
        constexpr auto owner = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
        table.at(split_page).host_extents = {{host, 6, 0, owner}, {host + 6, 4090, 6, owner}};
        auto serial_fault = std::make_shared<amdgpu::GraphicsDraw>(*draw);
        EXPECT_THROW(serial_fault->advance(*access), std::runtime_error);
        const auto prefix = storage;
        EXPECT_NE(prefix, initial);
        std::copy(initial.begin(), initial.end(), storage.begin());
        auto tiled_fault = std::make_shared<amdgpu::GraphicsDraw>(*draw);
        EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*tiled_fault, *access, pool));
        EXPECT_EQ(storage, initial);
        EXPECT_THROW(tiled_fault->advance(*access, &pool, 4), std::runtime_error);
        EXPECT_EQ(storage, prefix);
        table.at(split_page) = original_entry;
        std::copy(initial.begin(), initial.end(), storage.begin());
        auto invalid = std::make_shared<amdgpu::GraphicsDraw>(*draw);
        wave_->set_wg_coord(dispatch->total_wgs - 1, 0, 0);
        invalid->export_lane(*wave_, 0, 0, 4, {});
        EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*invalid, *access, pool));
        EXPECT_EQ(storage, initial);
        EXPECT_THROW(invalid->advance(*access), std::runtime_error);
        EXPECT_NE(storage, initial);
        std::copy(initial.begin(), initial.end(), storage.begin());
        std::fenv_t masked;
        std::fegetenv(&masked);
        std::feclearexcept(FE_INVALID);
        feenableexcept(FE_INVALID);
        EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
        std::fesetenv(&masked);
      }
      std::fesetround(rounding);
      std::feclearexcept(FE_ALL_EXCEPT);
      std::feraiseexcept(FE_DIVBYZERO);
      std::fenv_t before;
      std::fegetenv(&before);
      before.__mxcsr |= 2; // Preserve the denormal-operand flag as well.
      std::fesetenv(&before);
      errno = E2BIG;
      ASSERT_TRUE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
      std::fenv_t parallel_environment;
      std::fegetenv(&parallel_environment);
      const int parallel_errno = errno;
      const auto parallel = storage;
      std::copy(initial.begin(), initial.end(), storage.begin());
      std::fesetenv(&before);
      errno = E2BIG;
      EXPECT_FALSE(draw->advance(*access));
      std::fenv_t serial_environment;
      std::fegetenv(&serial_environment);
      EXPECT_EQ(errno, parallel_errno);
      EXPECT_EQ(serial_environment.__control_word, parallel_environment.__control_word);
      EXPECT_EQ(serial_environment.__status_word & 0x3f, parallel_environment.__status_word & 0x3f);
      EXPECT_EQ(serial_environment.__mxcsr, parallel_environment.__mxcsr);
      EXPECT_EQ(storage, parallel);
      EXPECT_NE(storage, initial);
    }
  }
#endif
}

TEST_P(GraphicsExportTest, ParallelDepthAndColorTilesPreserveComparisonsOverlapAndHostEnvironment) {
#if defined(__GLIBC__) && defined(__x86_64__)
  constexpr uint32_t width = 129, height = 65;
  constexpr uint64_t base = 0x100000;
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  amdgpu::LegacyPageTable table;
  util::DistributedSharedMutex table_mutex;
  auto request_mutex = std::make_shared<util::DistributedSharedMutex>();
  // Two mipmapped surfaces include their full nonzero array-slice strides.
  std::vector<uint8_t> storage(2 << 20);
  for (size_t page = 0; page < storage.size() / 4096; ++page)
    table[(base >> 12) + page] = {storage.data() + page * 4096, amdgpu::Mtype::RW,
                                  amdgpu::LegacyHostExtentOwner::DriverSealedRam};
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter(vm, &memory_);
  const auto handle =
      adapter.register_address_space(7, &table, &table_mutex, nullptr, request_mutex);
  auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  amdgpu::CpuDispatchPool pool(4);
  std::fenv_t saved_environment;
  std::fegetenv(&saved_environment);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved_environment, saved_errno};

  const auto prepare = [&](const amdgpu::Pm4QueueState &state, bool color_exports = true) {
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 9);
    // Three complete overlapping rectangles. Every pixel must observe primitive
    // order even when other tiles complete on different workers.
    for (uint32_t primitive = 0; primitive < 3; ++primitive)
      export_rectangle_vertices(*draw, primitive);
    const auto dispatch = draw->advance(*access);
    EXPECT_TRUE(dispatch);
    if (!dispatch)
      return std::shared_ptr<amdgpu::GraphicsDraw>{};
    EXPECT_GE(dispatch->total_wgs, 128u);
    // At pixel (0,0), the successive depths are 0.75, 0.25, 0.5. Other lanes
    // exercise signed zero, subnormals, infinities, NaNs and UNORM rounding edges.
    constexpr std::array<uint32_t, 16> depths{0x3f400000, 0x80000000, 0,          1,
                                              0x007fffff, 0x3effffff, 0x3f000001, 0x3f800000,
                                              0x3f800001, 0xbf800000, 0x7f800000, 0xff800000,
                                              0x7fc01234, 0x7f801234, 0x3f000000, 0x3eaaaaab};
    constexpr uint32_t first_depth[] = {0x3f400000, 0x3e800000, 0x3f000000};
    for (uint32_t workgroup = 0; workgroup < dispatch->total_wgs; ++workgroup) {
      wave_->set_wg_coord(workgroup, 0, 0);
      const uint32_t primitive = workgroup / (dispatch->total_wgs / 3);
      for (uint32_t lane = 0; lane < dispatch->kernel_wave_size; ++lane) {
        draw->export_lane(*wave_, lane, 8, 1,
                          {lane % depths.size() == 0 ? first_depth[primitive]
                                                     : depths[(lane + primitive) % depths.size()],
                           0, 0, 0});
        // Missing color must not suppress the middle primitive's depth write.
        if (color_exports && state.context_registers[gfx12 ? 0x214 : 0x8e] &&
            !(primitive == 1 && lane % 8 == 0)) {
          constexpr uint32_t rg[] = {0x30003a00, 0x3a003400, 0x34003800};
          draw->export_lane(*wave_, lane, 0, 3, {rg[primitive], 0x38003400, 0, 0});
        }
      }
    }
    return draw;
  };

  for (uint32_t attachment : {0u, 1u, 2u}) {
    const uint32_t bytes = attachment == 0 ? 2 : 4;
    const bool color = attachment == 2;
    // Linear pitch padding, a tiled surface, and a nonzero mip in one array
    // slice exercise the address preflight independently of the pixel math.
    for (uint32_t layout = 0; layout < 3; ++layout) {
      const uint32_t swizzle = layout == 0 ? 0 : gfx12 ? 3 : layout == 1 ? 24 : 28;
      const uint32_t level = layout == 2, first_layer = layout == 2;
      const auto mip = amdgpu::image_mip_layout(gfx12, swizzle, bytes, width << level,
                                                height << level, level + 1, level);
      ASSERT_TRUE(mip);
      const uint64_t layer_base = amdgpu::image_layer_base(
          gfx12, base + mip->offset, mip->slice_size, first_layer, bytes, swizzle);
      const auto first_address =
          gfx12 ? amdgpu::gfx12_image_address(layer_base, mip->tail_x, mip->tail_y, mip->pitch,
                                              bytes, swizzle)
                : amdgpu::gfx11_image_address(layer_base, mip->tail_x, mip->tail_y, mip->pitch,
                                              bytes, swizzle);
      ASSERT_TRUE(first_address);
      ASSERT_LE(*first_address + bytes, base + storage.size());
      for (uint32_t comparison = 0; comparison < 8; ++comparison) {
        // D16 always refuses parallel output. Keep one layout/comparison across
        // caller FP controls; its numeric comparisons and subresource layouts
        // have separate serial tests below.
        if (bytes == 2 && (layout != 0 || comparison != 7))
          continue;
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          SCOPED_TRACE(testing::Message()
                       << attachment << ',' << layout << ',' << comparison << ',' << rounding);
          std::fesetenv(FE_DFL_ENV);
          auto state = rectangle_state();
          auto &ctx = state.context_registers;
          ctx[gfx12 ? 5 : 7] = ((width << level) - 1) | (((height << level) - 1) << 16);
          ctx[gfx12 ? 6 : 0x10] =
              (bytes == 2 ? 1 : 3) | (swizzle << 4) | (level << (gfx12 ? 15 : 16));
          if (gfx12) {
            ctx[1] = first_layer | (first_layer << 16);
            ctx[2] = level << 26;
            ctx[8] = ctx[10] = base >> 8;
          } else {
            ctx[2] = first_layer | (first_layer << 13) | (level << 26);
            ctx[0x12] = ctx[0x14] = base >> 8;
          }
          ctx[gfx12 ? 0x1c : 0x200] = 6 | (comparison << 4);
          ctx[gfx12 ? 0x19 : 3] = rounding == FE_TOWARDZERO ? (gfx12 ? 1u : 1u << 16) : 0;
          ctx[gfx12 ? 0x115 : 0xb4] = std::bit_cast<uint32_t>(0.125f);
          ctx[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(0.875f);
          ctx[0x10f] = ctx[0x110] = std::bit_cast<uint32_t>(float(width) / 2);
          ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(float(height) / 2);
          ctx[0x91] = (width - gfx12) | ((height - gfx12) << 16);
          if (color) {
            // A separate R8 or RGBA8 surface has its own pitch and mip layout.
            ctx[0x318] = 0x1800;
            ctx[gfx12 ? 0x3b0 : 0x31c] = layout == 1 ? 1 : 10;
            ctx[gfx12 ? 0x319 : 0x31b] =
                first_layer | (first_layer << (gfx12 ? 14 : 13)) | (gfx12 ? 0 : level << 26);
            if (gfx12)
              ctx[0x31a] = level;
            ctx[gfx12 ? 0x31e : 0x3b0] =
                gfx12 ? (((width << level) - 1) << 16) | ((height << level) - 1)
                      : (((width << level) - 1) << 14) | ((height << level) - 1) | (level << 28);
            ctx[gfx12 ? 0x31f : 0x3b8] =
                first_layer | (swizzle << (gfx12 ? 15 : 14)) | (gfx12 ? level << 19 : 0);
            ctx[gfx12 ? 0x214 : 0x8e] = layout == 1 ? 1 : layout == 2 ? 5 : 15;
            ctx[gfx12 ? 0x215 : 0x8f] = layout == 1 ? 1 : 15;
            ctx[gfx12 ? 0x195 : 0x1c5] = 4;
            ctx[gfx12 ? 0x216 : 0x202] = rounding == FE_TOWARDZERO ? 0x660010 : 0xcc0010;
            ctx[0x1e0] =
                rounding == FE_DOWNWARD || rounding == FE_UPWARD ? (1u << 30) | 4u | (5u << 8) : 0;
          }
          auto draw = prepare(state);
          ASSERT_TRUE(draw);
          const uint32_t initial_bits = bytes == 2 ? 32768 : 0x3f000000;
          for (size_t offset = 0; offset < storage.size(); offset += bytes)
            std::memcpy(storage.data() + offset, &initial_bits, bytes);
          const auto initial = storage;

          if (!color && bytes == 4 && layout == 0 && comparison == 7 && rounding == FE_TONEAREST) {
            // Read-only depth still evaluates comparisons, but commits no bytes.
            auto readonly_state = rectangle_state();
            readonly_state.context_registers = ctx;
            readonly_state.context_registers[gfx12 ? 0x1c : 0x200] &= ~4u;
            auto readonly_draw = prepare(readonly_state);
            ASSERT_TRUE(readonly_draw);
            EXPECT_TRUE(
                amdgpu::GraphicsDrawTestAccess::parallel_output(*readonly_draw, *access, pool));
            EXPECT_EQ(storage, initial);
            for (uint32_t refusal = 0; refusal < 3; ++refusal) {
              auto refused_state = rectangle_state();
              refused_state.context_registers = ctx;
              auto &refused = refused_state.context_registers;
              if (refusal == 0) {
                refused[gfx12 ? 0x1c : 0x200] |= 1; // Active stencil is another surface.
                refused[gfx12 ? 7 : 0x11] = 1;
                refused[gfx12 ? 0xc : 0x13] = refused[gfx12 ? 0xe : 0x15] = 0x1800;
              } else if (refusal == 1) {
                refused[gfx12 ? 1 : 2] |= 1u << (gfx12 ? 16 : 13); // More than one layer.
              } else {
                // D16 plus color retains the serial conversion and its host flags.
                refused[gfx12 ? 6 : 0x10] = 1;
                refused[0x318] = 0x1800;
                refused[gfx12 ? 0x3b0 : 0x31c] = 10;
                refused[gfx12 ? 0x31e : 0x3b0] =
                    gfx12 ? ((width - 1) << 16) | (height - 1) : ((width - 1) << 14) | (height - 1);
                refused[gfx12 ? 0x214 : 0x8e] = refused[gfx12 ? 0x215 : 0x8f] = 15;
                refused[gfx12 ? 0x195 : 0x1c5] = 4;
                refused[gfx12 ? 0x216 : 0x202] = 0xcc0010;
              }
              auto refused_draw = prepare(refused_state);
              ASSERT_TRUE(refused_draw);
              EXPECT_FALSE(
                  amdgpu::GraphicsDrawTestAccess::parallel_output(*refused_draw, *access, pool));
              EXPECT_EQ(storage, initial);
            }
            auto invalid_state = rectangle_state();
            invalid_state.context_registers = ctx;
            invalid_state.context_registers[gfx12 ? 6 : 0x10] = 2; // Unsupported depth format.
            EXPECT_THROW(prepare(invalid_state), std::runtime_error);
            EXPECT_EQ(storage, initial);
            // Unknown backing and a later missing/aliased page must decline the
            // whole operation before any earlier pixel is changed.
            const uint64_t late_page = (base >> 12) + 2;
            const auto entry = table.at(late_page);
            table.at(late_page).host_extents.front().owner = amdgpu::LegacyHostExtentOwner::Driver;
            EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
            EXPECT_EQ(storage, initial);
            table.at(late_page) = entry;
            table.at(late_page).host_extents.front().host_ptr = storage.data();
            EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
            EXPECT_EQ(storage, initial);
            table.erase(late_page);
            EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
            EXPECT_EQ(storage, initial);
            auto faulting = std::make_shared<amdgpu::GraphicsDraw>(*draw);
            EXPECT_THROW(faulting->advance(*access), std::runtime_error);
            EXPECT_NE(storage, initial);
            table[late_page] = entry;
            std::copy(initial.begin(), initial.end(), storage.begin());
            // Whole-span contiguity does not make a split D32 read legal.
            auto *host = entry.host_extents.front().host_ptr;
            constexpr auto owner = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
            table.at(late_page).host_extents = {{host, 2, 0, owner}, {host + 2, 4094, 2, owner}};
            EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
            EXPECT_EQ(storage, initial);
            auto split_fault = std::make_shared<amdgpu::GraphicsDraw>(*draw);
            EXPECT_THROW(split_fault->advance(*access), std::runtime_error);
            EXPECT_NE(storage, initial);
            table.at(late_page) = entry;
            std::copy(initial.begin(), initial.end(), storage.begin());
            std::fenv_t masked;
            std::fegetenv(&masked);
            std::feclearexcept(FE_INVALID);
            feenableexcept(FE_INVALID);
            EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
            std::fesetenv(&masked);
            EXPECT_EQ(storage, initial);
          }

          if (color && layout == 0 && comparison == 7 && rounding == FE_TONEAREST) {
            // If every color export is absent, only the depth range is leased.
            auto no_color = prepare(state, false);
            ASSERT_TRUE(no_color);
            ASSERT_TRUE(amdgpu::GraphicsDrawTestAccess::parallel_output(*no_color, *access, pool));
            const auto depth_only = storage;
            EXPECT_NE(depth_only, initial);
            EXPECT_TRUE(
                std::equal(storage.begin() + 0x80000, storage.end(), initial.begin() + 0x80000));
            std::copy(initial.begin(), initial.end(), storage.begin());
            EXPECT_FALSE(no_color->advance(*access));
            EXPECT_EQ(storage, depth_only);
            std::copy(initial.begin(), initial.end(), storage.begin());
            auto alias_state = rectangle_state();
            alias_state.context_registers = ctx;
            alias_state.context_registers[0x318] = base >> 8;
            auto alias_draw = prepare(alias_state);
            ASSERT_TRUE(alias_draw);
            EXPECT_FALSE(
                amdgpu::GraphicsDrawTestAccess::parallel_output(*alias_draw, *access, pool));
            EXPECT_EQ(storage, initial);
            const auto original_table = table;
            for (uint64_t page = 0; page < 16; ++page)
              table[0x180 + page] = table.at(0x100 + page);
            EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
            EXPECT_EQ(storage, initial);
            table = original_table;
            auto mrt_state = rectangle_state();
            mrt_state.context_registers = ctx;
            auto &mrt = mrt_state.context_registers;
            mrt[gfx12 ? 0x321 : 0x327] = 0x1c00;
            mrt[gfx12 ? 0x3b1 : 0x32b] = 10;
            mrt[gfx12 ? 0x327 : 0x3b1] = mrt[gfx12 ? 0x31e : 0x3b0];
            mrt[gfx12 ? 0x214 : 0x8e] |= 0xf0;
            mrt[gfx12 ? 0x215 : 0x8f] |= 0xf0;
            mrt[gfx12 ? 0x195 : 0x1c5] |= 4 << 4;
            auto mrt_draw = prepare(mrt_state);
            ASSERT_TRUE(mrt_draw);
            EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*mrt_draw, *access, pool));
            EXPECT_EQ(storage, initial);
            for (uint64_t late_page : {uint64_t{0x102}, uint64_t{0x182}}) {
              const auto entry = table.at(late_page);
              table.erase(late_page);
              EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
              EXPECT_EQ(storage, initial);
              auto faulting = std::make_shared<amdgpu::GraphicsDraw>(*draw);
              EXPECT_THROW(faulting->advance(*access), std::runtime_error);
              EXPECT_NE(storage, initial);
              table[late_page] = entry;
              std::copy(initial.begin(), initial.end(), storage.begin());
            }
            // A first color access failure still follows that pixel's depth write.
            const auto entry = table.at(0x180);
            table.erase(0x180);
            EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::parallel_output(*draw, *access, pool));
            EXPECT_EQ(storage, initial);
            auto faulting = std::make_shared<amdgpu::GraphicsDraw>(*draw);
            EXPECT_THROW(faulting->advance(*access), std::runtime_error);
            uint32_t first_depth = 0;
            std::memcpy(&first_depth, storage.data(), 4);
            EXPECT_EQ(first_depth, 0x3f400000u);
            EXPECT_TRUE(std::equal(storage.begin() + 4, storage.end(), initial.begin() + 4));
            table[0x180] = entry;
            std::copy(initial.begin(), initial.end(), storage.begin());
          }

          std::fesetround(rounding);
          std::feclearexcept(FE_ALL_EXCEPT);
          std::feraiseexcept(FE_DIVBYZERO);
          std::fenv_t before;
          std::fegetenv(&before);
          before.__mxcsr |= 2;
          if (rounding == FE_TOWARDZERO)
            before.__mxcsr |= 0x8040; // DAZ and FTZ are independent caller controls.
          std::fesetenv(&before);
          errno = E2BIG;
          auto parallel_draw = std::make_shared<amdgpu::GraphicsDraw>(*draw);
          if (bytes == 4) {
            ASSERT_TRUE(
                amdgpu::GraphicsDrawTestAccess::parallel_output(*parallel_draw, *access, pool));
          } else {
            // D16 stays serial: compiler vectorization of the conversion can
            // otherwise change host exception flags in unused SIMD lanes.
            EXPECT_FALSE(
                amdgpu::GraphicsDrawTestAccess::parallel_output(*parallel_draw, *access, pool));
            EXPECT_EQ(storage, initial);
            EXPECT_FALSE(parallel_draw->advance(*access, &pool, 4));
          }
          std::fenv_t parallel_environment;
          std::fegetenv(&parallel_environment);
          const int parallel_errno = errno;
          const auto parallel = storage;
          uint32_t first = 0;
          std::memcpy(&first, storage.data() + *first_address - base, bytes);
          constexpr uint32_t expected32[] = {0x3f000000, 0x3e800000, 0x3f000000, 0x3e800000,
                                             0x3f400000, 0x3f000000, 0x3f400000, 0x3f000000};
          constexpr uint32_t expected16[] = {32768, 16384, 32768, 16384,
                                             49151, 32768, 49151, 32768};
          EXPECT_EQ(first, bytes == 2 ? expected16[comparison] : expected32[comparison]);
          if (color && layout == 0 && comparison == 7 && rounding == FE_TONEAREST) {
            EXPECT_EQ(storage[0x80000], 128u);
          }
          std::copy(initial.begin(), initial.end(), storage.begin());
          std::fesetenv(&before);
          errno = E2BIG;
          EXPECT_FALSE(draw->advance(*access));
          std::fenv_t serial_environment;
          std::fegetenv(&serial_environment);
          EXPECT_EQ(errno, parallel_errno);
          EXPECT_EQ(serial_environment.__control_word, parallel_environment.__control_word);
          EXPECT_EQ(serial_environment.__status_word & 0x3f,
                    parallel_environment.__status_word & 0x3f);
          EXPECT_EQ(serial_environment.__mxcsr, parallel_environment.__mxcsr);
          EXPECT_EQ(storage, parallel);
          if (comparison == 0 || (!color && comparison == 2)) {
            EXPECT_EQ(storage, initial);
          } else {
            EXPECT_NE(storage, initial);
          }
        }
      }
    }
  }
#endif
}

TEST_P(GraphicsExportTest, UncoveredQuadPreservesInterpolationExceptions) {
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (bool samples : {false, true}) {
    ASSERT_EQ(std::fesetenv(FE_DFL_ENV), 0);
    auto state = rectangle_state();
    const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
    state.uconfig_registers[0x242] = 4; // Triangle list.
    auto &ctx = state.context_registers;
    ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(0.5f);
    ctx[0x113] = std::bit_cast<uint32_t>(1.0f);
    ctx[0x91] = (4 - gfx12) | ((4 - gfx12) << 16);
    ctx[0x30e] = ctx[0x30f] = samples ? 0xffffffff : 0;
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    constexpr float xy[3][2] = {{-1, -1}, {1, -1}, {-1, 1}};
    for (uint32_t vertex = 0; vertex < 3; ++vertex) {
      const float w = vertex == 0 ? 1.0f : 0x1p-127f;
      draw->export_lane(*wave_, vertex, 12, 15,
                        {std::bit_cast<uint32_t>(xy[vertex][0] * w),
                         std::bit_cast<uint32_t>(xy[vertex][1] * w), 0,
                         std::bit_cast<uint32_t>(w)});
    }
    const uint32_t shift = gfx12 ? 9 : 10;
    draw->export_lane(*wave_, 0, 20, 1, {(1u << shift) | (2u << (2 * shift)), 0, 0, 0});
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    const auto dispatch = draw->advance(*access_);
    const int flags = std::fetestexcept(FE_ALL_EXCEPT);
    EXPECT_FALSE(dispatch);
    // No pixel center is covered, but the original interpolation still runs.
    // This literal witness also catches two paths agreeing after both skip it.
    EXPECT_EQ(flags, FE_OVERFLOW | FE_UNDERFLOW | FE_INEXACT);
  }
}

TEST_P(GraphicsExportTest, ParallelRasterPreservesWavePackingAndCallbackEnvironment) {
#if defined(__GLIBC__) && defined(__x86_64__)
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t width = 129, height = 97;
  // Record the public memory boundary, including status raised by the previous
  // primitive. A late refusal checks that the same fragment prefix exists.
  class RecordingMemory final : public amdgpu::PhysicalMemoryAccess {
  public:
    amdgpu::GpuMemoryPhysicalAccess backing;
    std::vector<std::array<uint64_t, 5>> reads;
    size_t fail_at = SIZE_MAX;
    explicit RecordingMemory(amdgpu::GpuMemory &memory) : backing(memory) {}
    amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain domain, uint64_t address,
                                 std::span<std::byte> bytes) override {
      std::fenv_t environment;
      std::fegetenv(&environment);
      reads.push_back({address, environment.__control_word,
                       uint64_t(environment.__status_word & 0x3f), environment.__mxcsr,
                       uint64_t(errno)});
      if (reads.size() == fail_at)
        return amdgpu::VmAccessOutcome::Faulted;
      return backing.read(domain, address, bytes);
    }
    amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain domain, uint64_t address,
                                  std::span<const std::byte> bytes) override {
      return backing.write(domain, address, bytes);
    }
  };
  auto recording = std::make_shared<RecordingMemory>(memory_);
  amdgpu::GpuVm vm;
  const auto handle = vm.register_address_space(
      0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(), recording, {}, true);
  auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  amdgpu::CpuDispatchPool pool(4);
  std::fenv_t saved;
  std::fegetenv(&saved);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  // Give every persistent worker a distinct environment from the caller. The
  // barrier forces all four pool threads to participate in these checks.
  std::barrier worker_barrier(4);
  pool.run_indexed(4, 4, [&](size_t) {
    std::fesetenv(FE_DFL_ENV);
    std::fesetround(FE_DOWNWARD);
    std::feraiseexcept(FE_OVERFLOW);
    errno = EILSEQ;
    worker_barrier.arrive_and_wait();
  });
  std::fenv_t worker_environment;
  std::fegetenv(&worker_environment);
  for (uint32_t kind : {0u, 1u, 2u}) {
    for (bool wave32 : {false, true}) {
      for (const auto &[rounding, traps] :
           {std::pair{FE_TONEAREST, false}, std::pair{FE_DOWNWARD, false},
            std::pair{FE_UPWARD, false}, std::pair{FE_TOWARDZERO, false},
            std::pair{FE_TONEAREST, true}}) {
        SCOPED_TRACE(kind);
        SCOPED_TRACE(wave32);
        SCOPED_TRACE(rounding);
        SCOPED_TRACE(traps);
        std::fesetenv(FE_DFL_ENV);
        auto state = rectangle_state();
        state.uconfig_registers[0x242] = kind == 0 ? 17 : 4;
        auto &ctx = state.context_registers;
        ctx[0x10f] = ctx[0x110] = std::bit_cast<uint32_t>(float(width) / 2);
        ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(float(height) / 2);
        ctx[0x113] = std::bit_cast<uint32_t>(0.75f);
        ctx[0x114] = std::bit_cast<uint32_t>(0.125f);
        ctx[0x90] = 1 | (3 << 16);
        ctx[0x91] = (width - gfx12) | ((height - gfx12) << 16);
        ctx[0x30f] = 1; // Half-covered quads retain helper interpolation.
        ctx[gfx12 ? 0x190 : 0x1b6] = (wave32 ? 1u << 15 : 0) | 1;
        ctx[gfx12 ? 0x10b : 0x2fa] = ctx[gfx12 ? 0x10d : 0x2fc] = std::bit_cast<uint32_t>(1.0f);
        if (gfx12)
          state.sh_registers[0x31] = 1u << 11;
        ctx[gfx12 ? 0x199 : 0x191] = 0;
        state.uconfig_registers[0x446] = 0x12340;
        auto original = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 6);
        for (uint32_t primitive = 0; primitive < 2; ++primitive) {
          for (uint32_t vertex = 0; vertex < 3; ++vertex) {
            const uint32_t index = primitive * 3 + vertex;
            const float w = kind == 1 ? float(1u << vertex) : 1.0f;
            const float extent = kind == 2 ? 0.0625f : kind == 1 ? 1.125f : 1.0f;
            original->export_lane(*wave_, index, 12, 15,
                                  {std::bit_cast<uint32_t>((vertex == 2 ? extent : -extent) * w),
                                   std::bit_cast<uint32_t>((vertex == 1 ? extent : -extent) * w),
                                   std::bit_cast<uint32_t>((0.125f + vertex * 0.125f) * w),
                                   std::bit_cast<uint32_t>(w)});
            for (uint32_t component = 0; component < 4; ++component)
              memory_.write32(0x123400000ull + index * 16 + component * 4,
                              std::bit_cast<uint32_t>(float(index * 4 + component) / 7));
          }
          const uint32_t first = primitive * 3, shift = gfx12 ? 9 : 10;
          original->export_lane(
              *wave_, primitive, 20, 1,
              {first | ((first + 1) << shift) | ((first + 2) << (2 * shift)), 0, 0, 0});
        }
        std::fesetround(rounding);
        std::feraiseexcept(FE_DIVBYZERO);
        std::fenv_t before;
        std::fegetenv(&before);
        before.__mxcsr |= 2; // Include the non-standard denormal sticky flag.
        if (rounding == FE_TOWARDZERO)
          before.__mxcsr |= 0x8040; // Also retain DAZ/FTZ controls.
        if (traps) {
          before.__control_word &= ~1u;
          before.__mxcsr &= ~0x80u; // Unmasked invalid operations use the serial path.
        }
        for (bool fault : {false, true}) {
          recording->fail_at = fault ? 13 : SIZE_MAX;
          std::array<std::vector<uint32_t>, 2> bits;
          std::array<std::vector<std::array<uint64_t, 5>>, 2> reads;
          std::array<std::fenv_t, 2> after;
          std::array<int, 2> errors;
          std::array<uint32_t, 2> waves{};
          for (size_t parallel = 0; parallel < 2; ++parallel) {
            auto draw = std::make_shared<amdgpu::GraphicsDraw>(*original);
            // Reuse nonzero geometry storage so the comparison also witnesses
            // every live field overwrite and exact zero padding at both widths.
            if (parallel && !fault && kind == 0 && rounding == FE_TONEAREST && !traps)
              amdgpu::GraphicsDrawTestAccess::poison_retired_fragment_storage(*draw, 1024);
            recording->reads.clear();
            std::fesetenv(&before);
            errno = E2BIG;
            if (fault) {
              EXPECT_THROW(draw->advance(*access, parallel ? &pool : nullptr, 4),
                           std::runtime_error);
            } else {
              auto dispatch = draw->advance(*access, parallel ? &pool : nullptr, 4);
              ASSERT_TRUE(dispatch);
              waves[parallel] = dispatch->total_wgs;
            }
            std::fegetenv(&after[parallel]);
            errors[parallel] = errno;
            reads[parallel] = recording->reads;
            bits[parallel] = amdgpu::GraphicsDrawTestAccess::fragment_bits(*draw);
          }
          EXPECT_EQ(waves[0], waves[1]);
          EXPECT_EQ(bits[0], bits[1]);
          EXPECT_EQ(reads[0], reads[1]);
          EXPECT_EQ(errors[0], errors[1]);
          EXPECT_EQ(after[0].__control_word, after[1].__control_word);
          EXPECT_EQ(after[0].__status_word & 0x3f, after[1].__status_word & 0x3f);
          EXPECT_EQ(after[0].__mxcsr, after[1].__mxcsr);
          if (!fault && kind != 2) {
            EXPECT_GT(waves[0], 100u);
          }
        }
      }
    }
  }
  std::fesetenv(&worker_environment);
  errno = EILSEQ;
  // The pool/barrier may leave errno=EAGAIN after a futex wait between
  // submissions. Caller/callback errno is checked above; persistent worker
  // controls and sticky FP flags remain observable across these waits.
  std::array<std::array<uint32_t, 3>, 4> observed{};
  pool.run_indexed(4, 4, [&](size_t index) {
    std::fenv_t after;
    std::fegetenv(&after);
    observed[index] = {after.__control_word, uint32_t(after.__status_word & 0x3f), after.__mxcsr};
    worker_barrier.arrive_and_wait();
  });
  for (const auto &environment : observed)
    EXPECT_EQ(environment,
              (std::array<uint32_t, 3>{worker_environment.__control_word,
                                       uint32_t(worker_environment.__status_word & 0x3f),
                                       worker_environment.__mxcsr}));
#endif
}

TEST(GraphicsRasterMathTest, InterpolationMatchesPhysicalRdna4QuadInputs) {
  // Raw float bits captured with GL_AMD_shader_explicit_vertex_parameter.
  // Both ordinary and near-edge quads exercise opposite gradient signs and
  // cancellation during the shared-exponent addition of two pixel offsets.
  struct Case {
    double area, edge_x, edge_y, x, y;
    std::array<uint32_t, 4> expected;
  };
  const Case cases[] = {
      {33600, -160, 0, -2.5, -149.5, {0x3c430c30, 0x3bea0ea0, 0x3c430c30, 0x3bea0ea0}},
      {33600, 0, -210, -2.5, -149.5, {0x3f6f3332, 0x3f6f3332, 0x3f6d9999, 0x3f6d9999}},
      {31500, 40, -210, 119.5, -29.5, {0x3eb26324, 0x3eb30994, 0x3eaef954, 0x3eaf9fc4}},
      {31500, 120, 157.5, 119.5, -29.5, {0x3e9d8fd7, 0x3e9f8329, 0x3ea01f33, 0x3ea21285}},
      {31500, 40, -210, 5.5, -1.5, {0x3c8b224a, 0x3c958956, 0x3c290a8e, 0x3c3dd8a6}},
      {31500, 120, 157.5, 5.5, -1.5, {0x3c5c675e, 0x3c8d68d5, 0x3c972971, 0x3cb65e97}},
      {29400, 160, -52.5, 115.5, 80.5, {0x3ef83a82, 0x3efb03d3, 0x3ef75074, 0x3efa19c5}},
      {29400, -80, 210, 115.5, 80.5, {0x3e857c56, 0x3e8417ae, 0x3e892490, 0x3e87bfe8}},
  };
  for (const auto &test : cases) {
    const float inverse_area = amdgpu::raster::truncate_float(1.0 / test.area);
    const amdgpu::raster::Plane plane{amdgpu::raster::truncate_float(test.edge_x * inverse_area),
                                      amdgpu::raster::truncate_float(test.edge_y * inverse_area)};
    for (uint32_t lane = 0; lane < 4; ++lane)
      EXPECT_EQ(std::bit_cast<uint32_t>(plane.at_quad(test.x, test.y, lane)), test.expected[lane]);
  }
}

namespace {

// Keep the prior libm expression as a differential oracle. Function boundaries
// prevent callers from folding arithmetic across the host-environment checks.
[[gnu::noinline]] float original_quad_offsets(float center, float dx, float dy) {
  const float largest = std::max({std::abs(center), std::abs(dx), std::abs(dy)});
  if (largest == 0)
    return 0;
  const double unit = std::ldexp(1.0, std::ilogb(largest) - 23);
  return amdgpu::raster::truncate_float(
      (std::trunc(center / unit) + std::trunc(dx / unit) + std::trunc(dy / unit)) * unit);
}

[[gnu::noinline]] float current_quad_offsets(float center, float dx, float dy) {
  return amdgpu::raster::add_quad_offsets(center, dx, dy);
}

} // namespace

TEST(GraphicsRasterMathTest, QuadOffsetUnitsMatchLibmAcrossFiniteExponentsAndHostModes) {
  std::vector<std::array<uint32_t, 3>> inputs;
  const auto append = [&](uint32_t magnitude) {
    inputs.push_back({magnitude, magnitude ^ 0x80000000u, 1});
    inputs.push_back({magnitude | 0x80000000u, magnitude, magnitude | 0x80000000u});
    inputs.push_back({magnitude, 0, magnitude - 1});
  };
  for (uint32_t exponent = 1; exponent < 255; ++exponent)
    for (uint32_t mantissa : {0u, 1u, 0x3fffffu, 0x7fffffu})
      append((exponent << 23) | mantissa);
  for (uint32_t bit = 0; bit < 23; ++bit) {
    append(1u << bit);
    append((1u << (bit + 1)) - 1);
  }
  inputs.insert(inputs.end(), {{0, 0, 0},
                               {0x80000000, 0, 0x80000000},
                               {0x7f800000, 0, 0},
                               {0xff800000, 0x7f800000, 0},
                               {0x3f800000, 0x7fc12345, 0},
                               {0x3f800000, 0, 0xff801234}});
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    for (uint32_t denorm : {0u, 0x40u, 0x8000u, 0x8040u}) {
      ASSERT_EQ(std::fesetround(rounding), 0);
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
      std::fenv_t before;
      ASSERT_EQ(std::fegetenv(&before), 0);
#if defined(__GLIBC__) && defined(__x86_64__)
      before.__mxcsr = (before.__mxcsr & ~0x8040u) | denorm;
#else
      if (denorm)
        continue;
#endif
      for (const auto &input : inputs) {
        struct Result {
          uint32_t bits;
          int error, flags;
          std::fenv_t environment;
        };
        const auto run = [&](auto function) {
          std::fesetenv(&before);
          errno = EDOM;
          const uint32_t bits = std::bit_cast<uint32_t>(function(std::bit_cast<float>(input[0]),
                                                                 std::bit_cast<float>(input[1]),
                                                                 std::bit_cast<float>(input[2])));
          Result result{bits, errno, std::fetestexcept(FE_ALL_EXCEPT), {}};
          std::fegetenv(&result.environment);
          return result;
        };
        const auto original = run(original_quad_offsets);
        const auto current = run(current_quad_offsets);
        EXPECT_EQ(current.bits, original.bits);
        EXPECT_EQ(current.error, original.error);
        EXPECT_EQ(current.flags, original.flags);
#if defined(__GLIBC__) && defined(__x86_64__)
        EXPECT_EQ(current.environment.__control_word, original.environment.__control_word);
        EXPECT_EQ(current.environment.__status_word & 0x3f,
                  original.environment.__status_word & 0x3f);
        EXPECT_EQ(current.environment.__mxcsr, original.environment.__mxcsr);
#endif
      }
    }
  }
}

TEST(GraphicsRasterMathTest, QuadOffsetNanCentersKeepDomainReportingAndQuietPayloads) {
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    for (uint32_t denorm : {0u, 0x40u, 0x8000u, 0x8040u}) {
      ASSERT_EQ(std::fesetround(rounding), 0);
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO | FE_INEXACT), 0);
      std::fenv_t before;
      ASSERT_EQ(std::fegetenv(&before), 0);
#if defined(__GLIBC__) && defined(__x86_64__)
      before.__mxcsr = (before.__mxcsr & ~0x8040u) | denorm;
#else
      if (denorm)
        continue;
#endif
      for (uint32_t center : {0x7fc12345u, 0xffc54321u, 0x7f800001u, 0xff812345u}) {
        for (const auto &offsets :
             {std::array{0u, 0x80000000u}, std::array{0x3f800000u, 0xbf800000u},
              std::array{0x7f800000u, 0xff800000u}, std::array{0x7f800123u, 0xffc45678u}}) {
          ASSERT_EQ(std::fesetenv(&before), 0);
          errno = EINVAL;
          const uint32_t bits = std::bit_cast<uint32_t>(
              current_quad_offsets(std::bit_cast<float>(center), std::bit_cast<float>(offsets[0]),
                                   std::bit_cast<float>(offsets[1])));
          const int error = errno;
          const int flags = std::fetestexcept(FE_ALL_EXCEPT);
          std::fenv_t after;
          ASSERT_EQ(std::fegetenv(&after), 0);
          EXPECT_EQ(bits, center | 0x00400000u);
          EXPECT_EQ(error, (math_errhandling & MATH_ERRNO) ? EDOM : EINVAL);
          EXPECT_EQ(flags, FE_INVALID | FE_DIVBYZERO | FE_INEXACT);
          EXPECT_EQ(std::fegetround(), rounding);
#if defined(__GLIBC__) && defined(__x86_64__)
          EXPECT_EQ(after.__control_word, before.__control_word);
          EXPECT_EQ(after.__mxcsr & ~0x3fu, before.__mxcsr & ~0x3fu);
          EXPECT_EQ(after.__status_word & before.__status_word & 0x3fu,
                    before.__status_word & 0x3fu);
          EXPECT_EQ(after.__mxcsr & before.__mxcsr & 0x3fu, before.__mxcsr & 0x3fu);
#endif
        }
      }
    }
  }
}

TEST(GraphicsRasterMathTest, Sse41PlanesRetainResultAndHostEnvironment) {
#if defined(__clang__) && defined(__x86_64__)
  if (!amdgpu::raster::supports_sse41_planes()) {
    GTEST_SKIP() << "SSE4.1 plane evaluation is unavailable";
  }
  const std::array<uint32_t, 3> inputs[] = {
      {0, 0, 0},
      {0x80000000, 0, 0x80000000},
      {0x3e800000, 0xbe800000, 0x3f800001},
      {1, 0x80000001, 0x007fffff},
      {0x00800000, 0x807fffff, 1},
      {0x7f7fffff, 1, 0xff7fffff},
      {0x7fc12345, 0xff812345, 0x807fffff},
      {0x3f800000, 0xbf800000, 0xffc54321},
      {0x7f800000, 0x7fc12345, 0},
      {0, 0x7f800000, 0xff800000},
  };
  const std::array<double, 2> coordinates[] = {{0, 0}, {0.5, -0.5}, {-0x1.acp+5, 0x1.b6p+6}};
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    for (uint32_t denorm : {0u, 0x40u, 0x8000u, 0x8040u}) {
      for (int sticky : {0, FE_DIVBYZERO}) {
        ASSERT_EQ(std::fesetenv(FE_DFL_ENV), 0);
        ASSERT_EQ(std::fesetround(rounding), 0);
        ASSERT_EQ(std::feraiseexcept(sticky), 0);
        std::fenv_t before;
        ASSERT_EQ(std::fegetenv(&before), 0);
#if defined(__GLIBC__)
        before.__mxcsr = (before.__mxcsr & ~0x8040u) | denorm;
#else
        if (denorm)
          continue;
#endif
        for (const auto &input : inputs) {
          const amdgpu::raster::Plane plane{std::bit_cast<float>(input[0]),
                                            std::bit_cast<float>(input[1]),
                                            std::bit_cast<float>(input[2])};
          for (const auto &xy : coordinates) {
            for (uint32_t lane = 0; lane < 4; ++lane) {
              struct Result {
                uint32_t bits;
                int error, flags;
                std::fenv_t environment;
              };
              const auto run = [&](bool sse41) {
                std::fesetenv(&before);
                errno = E2BIG;
                const float value = sse41 ? plane.at_quad_sse41(xy[0], xy[1], lane)
                                          : plane.at_quad(xy[0], xy[1], lane);
                Result result{
                    std::bit_cast<uint32_t>(value), errno, std::fetestexcept(FE_ALL_EXCEPT), {}};
                std::fegetenv(&result.environment);
                return result;
              };
              const auto original = run(false), targeted = run(true);
              EXPECT_EQ(targeted.bits, original.bits);
              EXPECT_EQ(targeted.error, original.error);
              EXPECT_EQ(targeted.flags, original.flags);
#if defined(__GLIBC__)
              EXPECT_EQ(targeted.environment.__control_word, original.environment.__control_word);
              EXPECT_EQ(targeted.environment.__status_word & 0x3f,
                        original.environment.__status_word & 0x3f);
              EXPECT_EQ(targeted.environment.__mxcsr, original.environment.__mxcsr);
#endif
            }
          }
        }
      }
    }
  }
#else
  EXPECT_FALSE(amdgpu::raster::supports_sse41_planes());
#endif
}

TEST(GraphicsRasterMathTest, BoundedPlaneAdmissionRetainsRawHostState) {
#if defined(__GLIBC__) && defined(__x86_64__)
  using amdgpu::raster::Plane;
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  std::fesetenv(FE_DFL_ENV);
  std::fenv_t masked;
  std::fegetenv(&masked);
  masked.__mxcsr |= 0x20;
  const auto check = [&](const Plane &plane, double origin, bool bounded) {
    for (uint32_t mode = 0; mode < 5; ++mode) {
      auto before = masked;
      if (mode == 1)
        before.__mxcsr &= ~0x20u; // INEXACT is not sticky yet.
      if (mode == 2)
        before.__mxcsr &= ~0x80u; // SSE invalid trap enabled.
      if (mode >= 3)
        before.__control_word &= ~4u; // x87 divide-by-zero trap enabled.
      if (mode == 4)
        before.__status_word |= 0x8084u; // Pending x87 exception, still nonwaiting.
      std::fesetenv(&before);
      std::fegetenv(&before); // Snapshot the installed hardware representation.
      errno = E2BIG;
      const bool admitted = amdgpu::raster::can_omit_bounded_planes(plane, plane, origin, -0.0);
      const int error = errno;
      std::fenv_t after;
      std::fegetenv(&after);
      std::fesetenv(FE_DFL_ENV); // Clear pending traps before assertion diagnostics.
      EXPECT_EQ(admitted, bounded && mode == 0 && amdgpu::raster::supports_sse41_planes());
      EXPECT_EQ(error, E2BIG);
      EXPECT_EQ(after.__control_word, before.__control_word);
      EXPECT_EQ(after.__status_word, before.__status_word);
      EXPECT_EQ(after.__tags, before.__tags);
      EXPECT_EQ(after.__eip, before.__eip);
      EXPECT_EQ(after.__opcode, before.__opcode);
      EXPECT_EQ(after.__data_offset, before.__data_offset);
      EXPECT_EQ(after.__mxcsr, before.__mxcsr);
    }
  };
  for (const auto &[bits, bounded] : {std::pair{0u, true},
                                      {0x80000000u, true},
                                      {0x30800000u, true},
                                      {0xb0800000u, true},
                                      {0x4e800000u, true},
                                      {0xce800000u, true},
                                      {0x307fffffu, false},
                                      {0x4e800001u, false},
                                      {1u, false},
                                      {0x7f800000u, false},
                                      {0x7fc12345u, false},
                                      {0x7f812345u, false}}) {
    const float value = std::bit_cast<float>(bits);
    check({value, 0, 0}, 0, bounded);
    check({0, value, 0}, 0, bounded);
    check({0, 0, value}, 0, bounded);
  }
  for (const auto &[bits, bounded] : {std::pair{uint64_t{0}, true},
                                      {uint64_t{0x8000000000000000}, true},
                                      {uint64_t{0x3f70000000000000}, true},
                                      {uint64_t{0xbf70000000000000}, true},
                                      {uint64_t{0x41d0000000000000}, true},
                                      {uint64_t{0xc1d0000000000000}, true},
                                      {uint64_t{0x3f60000000000000}, false},
                                      {uint64_t{0x3ff0000000000001}, false},
                                      {uint64_t{0x41d0000000000001}, false},
                                      {uint64_t{1}, false},
                                      {uint64_t{0x7ff0000000000000}, false},
                                      {uint64_t{0x7ff0000000000001}, false}})
    check({1, -1, 0}, std::bit_cast<double>(bits), bounded);
#else
  EXPECT_FALSE(amdgpu::raster::can_omit_bounded_planes({}, {}, 0, 0));
#endif
}

TEST(GraphicsRasterMathTest, BoundedPlaneOmissionKeepsAllRoundingAndFlushModes) {
#if defined(__clang__) && defined(__GLIBC__) && defined(__x86_64__)
  using amdgpu::raster::Plane;
  if (!amdgpu::raster::supports_sse41_planes())
    GTEST_SKIP();
  const Plane planes[] = {{0, -0.0f, 0},
                          {0x1p-30f, -0x1p-30f, 0x1p-30f},
                          {0x1.000002p-30f, -0x1.fffffep-30f, -0x1p-30f},
                          {0x1p+30f, -0x1p+30f, 0x1p+30f},
                          {0x1p+30f, 0x1p-30f, -0x1p+30f},
                          {0.25f, -0.25f, 0x1.000002p+0f}};
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    for (uint32_t flush : {0u, 0x40u, 0x8000u, 0x8040u}) {
      for (uint32_t sticky : {0x20u, 0x3fu}) {
        std::fesetenv(FE_DFL_ENV);
        std::fesetround(rounding);
        std::fenv_t before;
        std::fegetenv(&before);
        before.__mxcsr |= flush | sticky;
        for (const auto &plane : planes) {
          for (double origin : {0.0, -0x1p-8, 0x1p+30, -0x1p+30}) {
            for (int pixel : {INT32_MIN, -1, 0, INT32_MAX}) {
              const double dx = double(pixel) + 0.5 - origin;
              const double dy = double(pixel) + 0.5 + origin;
              for (uint32_t lane = 0; lane < 4; ++lane) {
                std::fesetenv(&before);
                errno = EDOM;
                const bool admitted =
                    amdgpu::raster::can_omit_bounded_planes(plane, plane, origin, -origin);
                // Volatile sinks retain both actual evaluations. The contract
                // concerns host state; the unused result itself is not read.
                volatile float first = plane.at_quad_sse41(dx, dy, lane);
                volatile float second = plane.at_quad_sse41(dx, dy, lane);
                (void)first;
                (void)second;
                const int error = errno;
                std::fenv_t after;
                std::fegetenv(&after);
                EXPECT_TRUE(admitted);
                EXPECT_EQ(error, EDOM);
                EXPECT_EQ(after.__control_word, before.__control_word);
                EXPECT_EQ(after.__status_word, before.__status_word);
                EXPECT_EQ(after.__tags, before.__tags);
                EXPECT_EQ(after.__eip, before.__eip);
                EXPECT_EQ(after.__opcode, before.__opcode);
                EXPECT_EQ(after.__data_offset, before.__data_offset);
                EXPECT_EQ(after.__mxcsr, before.__mxcsr);
              }
            }
          }
        }
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST(GraphicsRasterMathTest, AttributeDifferencesMatchPhysicalInterpolation) {
  // Raw RDNA3/4 outputs from an 8x8 right triangle. Independent controls cover
  // cancellation, operand alignment and normalization across both signs.
  struct Case {
    std::array<float, 3> values;
    int x, y;
    uint32_t expected;
  };
  const Case cases[] = {
      {{{0x1.f2addcp+15f, 0x1.8eee8ep-5f, -0x1.2bd482p+23f}}, 0, 0, 0xc908477fu},
      {{{0x1.c50a4ap+24f, 0x1.b9c84ep+16f, -0x1.f4596p+2f}}, 0, 1, 0x4ba9f1aau},
      {{{0x1.1f2552p-24f, -0x1.e7792ap-4f, 0x1.820c08p-38f}}, 0, 0, 0xbbf3bc16u},
      {{{0x1.b9a9c8p-11f, 0x1.d29e82p-34f, -0x1.b39ab6p+13f}}, 0, 0, 0xc459cd4eu},
      {{{0x1.2e2b76p+25f, 0x1.e556ap-25f, -0x1.98d974p+33f}}, 0, 0, 0xce442989u},
      {{{0x1.ff268ep-26f, 0x1.017be2p-25f, 0x1.018d58p-25f}}, 1, 0, 0x33000742u},
      {{{0x1.0086fcp-31f, 0x1.fd5f6ap-32f, 0x1.ff00c2p-32f}}, 0, 1, 0x30001c26u},
      {{{-0x1.ffe8fap-1f, -0x1.006c56p+0f, -0x1.01271p+0f}}, 0, 3, 0xbf80410du},
      {{{0x1.013ecep+62f, 0x1.ffba0ep+61f, 0x1.fe9378p+61f}}, 3, 0, 0x5e80425bu},
      {{{-0x1.d0093p+103f, 0x1.d0093p+103f, 0x1.d0092ep+103f}}, 3, 0, 0xe6000000u},
      {{{-0x1.e57b7p-4f, -0x1.e57b7p-28f, 0x1.e57b7p-28f}}, 0, 0, 0xbdd46602u},
      {{{-0x1.a69a9cp-111f, -0x1.ea8d5ep-106f, 0x1.ea7672p-104f}}, 0, 0, 0x09a0cd18u},
      {{{-0x1.0121ecp+97f, 0x1.0121ecp+97f, 0x1.0121eap+97f}}, 0, 0, 0xefc0d971u},
      {{{0x1.eabfp+80f, 0x1.eabfp+56f, -0x1.eabfp+56f}}, 1, 2, 0x67755f80u},
  };
  for (const auto &test : cases) {
    const float p10 = amdgpu::raster::attribute_difference(test.values[1], test.values[0]);
    const float p20 = amdgpu::raster::attribute_difference(test.values[2], test.values[0]);
    const float result =
        std::fma(p20, (test.y + 0.5f) / 8, std::fma(p10, (test.x + 0.5f) / 8, test.values[0]));
    EXPECT_EQ(std::bit_cast<uint32_t>(result), test.expected);
  }
}

TEST(GraphicsRasterMathTest, WeightedGradientsMatchPhysicalInterpolationPlanes) {
  // Gradients recovered from raw pull-model values across rotating cube faces.
  // The last two cases exercise cancellation in the reciprocal-W numerator.
  struct Case {
    double edge1, edge2, delta1, delta2;
    float inverse_area;
    uint32_t expected;
  };
  const Case cases[] = {
      {48.63671875, -55.8359375, 0x1.996342p-3, 0, -0x1.9c6a64p-12f, 0xbb7a9a10},
      {49.96484375, -54.25, 0x1.7b0c18p-3, 0, -0x1.7b9fap-12f, 0xbb5b698e},
      {49.96484375, -54.25, 0, 0x1.6edff4p-3, -0x1.7b9fap-12f, 0x3b66945d},
      {-28.57421875, 22.65234375, 0, 0x1.6e6aecp-3, -0x1.3dd3a6p-10f, 0xbba1033e},
      {42.9609375, 6.41796875, 0x1.b7dd2ap-3, 0, -0x1.413274p-12f, 0xbb393b23},
      {59.9140625, -67.0390625, 0, 0x1.683f6ap-3, -0x1.413274p-12f, 0x3b6cba88},
      {49.25, 1.0859375, -0x1.8a7f8p-9, -0x1.6ce818p-5, -0x1.2d795ap-12f, 0x386d1582},
      {48.63671875, -55.8359375, -0x1.52c248p-5, -0x1.35102p-4, -0x1.9c6a64p-12f, 0xba6304d6},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(amdgpu::raster::plane_gradient(
                  test.edge1, test.edge2, test.delta1, test.delta2, test.inverse_area)),
              test.expected);
}

TEST(GraphicsRasterMathTest, VertexDifferencesMatchPhysicalReciprocalW) {
  // Raw pull-model W values captured on both physical RDNA3 and RDNA4.
  // Unequal vertex W values distinguish FP32 truncation of the differences
  // from both wide subtraction and rounding to nearest before weighting.
  // The last two cases straddle the 26-exponent operand cutoff.
  struct Case {
    std::array<std::array<float, 3>, 3> positions; // Homogeneous X, Y, W.
    uint32_t pixel, expected;
  };
  const Case cases[] = {
      {{{{-0x1.e40cp-1f, 0x1.33a78ep-1f, 0x1.841e4p+0f},
         {-0x1.03b21ap+9f, 0x1.caffdp+7f, 0x1.16f4aep+9f},
         {-0x1.ea0b2ep-3f, 0x1.9fd112p+0f, 0x1.d14486p-1f}}},
       12,
       0x3e9e2303u},
      {{{{0x1.deb3f4p-1f, -0x1.630bccp-1f, 0x1.03deaap+1f},
         {-0x1.503afcp-4f, -0x1.f27acep-6f, 0x1.dc85d2p-5f},
         {-0x1.aaa884p+4f, 0x1.066592p+2f, 0x1.119ebcp+4f}}},
       4,
       0x40ca85bcu},
      {{{{0x1.08077ap+3f, 0x1.871c58p+4f, 0x1.3ed0c2p+4f},
         {0x1.644c9cp+7f, -0x1.cf06acp+3f, 0x1.9d3fccp+6f},
         {-0x1.46fbf4p+5f, -0x1.035e62p+5f, 0x1.625e1ap+4f}}},
       1,
       0x3cfed2a4u},
      {{{{-0x1.17d4c4p-11f, 0x1.228112p-6f, 0x1.f1ec8ap-6f},
         {0x1.d51e56p-8f, 0x1.a18582p-6f, 0x1.c210c8p-7f},
         {0x1.843394p-12f, -0x1.0bc392p-14f, 0x1.208b8cp-12f}}},
       14,
       0x43e43e50u},
      {{{{-1, -1, 1}, {0x1.8p24f, -0x1.8p24f, 0x1.8p24f}, {-1, 1, 1}}}, 2, 0x3ec00001u},
      {{{{-1, -1, 1}, {0x1.8p25f, -0x1.8p25f, 0x1.8p25f}, {-1, 1, 1}}}, 2, 0x3ec00000u},
  };
  for (const auto &test : cases) {
    struct Vertex {
      double x, y;
      float w, reciprocal_w;
    };
    std::array<Vertex, 3> vertices;
    for (uint32_t i = 0; i < 3; ++i) {
      const auto &p = test.positions[i];
      vertices[i] = {amdgpu::raster::viewport_coordinate(p[0], p[2], 2, 2),
                     amdgpu::raster::viewport_coordinate(p[1], p[2], 2, 2), p[2],
                     amdgpu::raster::reciprocal(p[2])};
    }
    const auto first = std::min_element(vertices.begin(), vertices.end(),
                                        [](Vertex a, Vertex b) { return a.w < b.w; });
    std::rotate(vertices.begin(), first, vertices.end());
    const auto &[a, b, c] = vertices;
    const double area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    const float inverse_area = amdgpu::raster::truncate_float(1.0 / area);
    const float delta_b = amdgpu::raster::vertex_difference(b.reciprocal_w, a.reciprocal_w);
    const float delta_c = amdgpu::raster::vertex_difference(c.reciprocal_w, a.reciprocal_w);
    const amdgpu::raster::Plane plane{
        amdgpu::raster::plane_gradient(c.y - a.y, a.y - b.y, delta_b, delta_c, inverse_area),
        amdgpu::raster::plane_gradient(a.x - c.x, b.x - a.x, delta_b, delta_c, inverse_area),
        a.reciprocal_w};
    const uint32_t x = test.pixel % 4, y = test.pixel / 4;
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at_quad((x & ~1u) + 0.5 - a.x, (y & ~1u) + 0.5 - a.y,
                                                    (x & 1) + 2 * (y & 1))),
              test.expected);
  }
}

TEST(GraphicsRasterMathTest, DepthTilesMatchPhysicalRdna3AndRdna4) {
  // Raw gl_FragCoord.z witnesses exercise unequal slopes, tile boundaries,
  // and negative tile-center depths in otherwise visible triangles. Power-of-two
  // spans separate evaluation precision from reciprocal-area approximation.
  struct Case {
    std::array<float, 3> z;
    double origin_x, origin_y, span_x, span_y;
    int x, y;
    uint32_t expected;
  };
  const Case cases[] = {
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 8, 8, 0, 0, 0x3d63c5fc},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 8, 8, 0, 2, 0x3e88c240},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 8, 8, 0, 0, 0x3d944e09},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 8, 8, 0, 2, 0x3ea2aa55},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 8, 8, 0, 0, 0x3d24cc50},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 8, 8, 0, 2, 0x3e311805},
      {{0x1.c589bcp-7f, 0x1.5c398cp-6f, 0x1.357a4ep-1f}, 0, 0, 8, 8, 0, 0, 0x3d51c91d},
      {{0x1.c589bcp-7f, 0x1.5c398cp-6f, 0x1.357a4ep-1f}, 0, 0, 8, 8, 0, 2, 0x3e4ba45a},
      {{0x1.28e624p-5f, 0x1.611d44p-1f, 0x1.56aff2p-7f}, 0, 0, 8, 8, 0, 0, 0x3d9a9058},
      {{0x1.28e624p-5f, 0x1.611d44p-1f, 0x1.56aff2p-7f}, 0, 0, 8, 8, 0, 2, 0x3d8d5cb6},
      {{0x1.efd10ap-5f, 0x1.9178ecp-1f, 0x1.aa8638p-4f}, 0, 0, 8, 8, 0, 0, 0x3dde2826},
      {{0x1.efd10ap-5f, 0x1.9178ecp-1f, 0x1.aa8638p-4f}, 0, 0, 8, 8, 0, 2, 0x3df47bdd},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 256, 256, 7, 7, 0x3cdb69da},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 256, 256, 8, 8, 0x3cf731a1},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 256, 256, 31, 32, 0x3de44db9},
      {{0x1.61f094p-10f, 0x1.80ed3p-6f, 0x1.b1d6fcp-1f}, 0, 0, 256, 256, 127, 127, 0x3ede10f3},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 256, 256, 7, 7, 0x3d19fafc},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 256, 256, 8, 8, 0x3d2ac2a4},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 256, 256, 31, 32, 0x3e0f1968},
      {{0x1.c218d2p-8f, 0x1.30ef8cp-4f, 0x1.f9df8p-1f}, 0, 0, 256, 256, 127, 127, 0x3f07791e},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 256, 256, 7, 7, 0x3cb63c42},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 256, 256, 8, 8, 0x3cc7926b},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 256, 256, 31, 32, 0x3d99d32f},
      {{0x1.a1b06ep-8f, 0x1.1b8234p-6f, 0x1.130d44p-1f}, 0, 0, 256, 256, 127, 127, 0x3e8d69f8},
      {{0x1.2a4644p-1f, 0x1.344512p-17f, 0x1.a61d1ep-17f}, 0, -0.75, 1, 8, 0, 0, 0x3e4d1209},
      {{0x1.908346p-12f, 0x1.d4e288p-12f, 0x1.1a08bcp-16f}, -1.25, 0.25, 8, 1, 0, 0, 0x399fdfb6},
      {{0x1.4e9ceap-16f, 0x1.81919ep-19f, 0x1.5d1c5p-21f}, -1.25, 1.5, 4, 1, 1, 1, 0x3709b38c},
  };
  for (const auto &test : cases) {
    const amdgpu::raster::DepthPlane plane{
        amdgpu::raster::vertex_difference(test.z[1], test.z[0]) / test.span_x,
        amdgpu::raster::vertex_difference(test.z[2], test.z[0]) / test.span_y, test.z[0],
        test.origin_x, test.origin_y};
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at(test.x, test.y)), test.expected);
  }
}

TEST(GraphicsRasterMathTest, AreaReciprocalMatchesPhysicalInterpolationBoundaries) {
  // Each inverse is isolated by all three raw pull-model planes on both cards.
  // Exact division followed by truncation misses each boundary by one FP32 step.
  struct Case {
    double area;
    uint32_t expected;
  };
  const Case cases[] = {
      {-0x1.cd58p+2, 0xbe0e0df9u},       {0x1.31b7a8p+5, 0x3cd65e2fu},
      {-0x1.82db3p+4, 0xbd296818u},      {0x1.bc544p+5, 0x3c937e89u},
      {0x1.4e50ep+5, 0x3cc407b8u},       {0x1.62182p+3, 0x3db914a7u},
      {0x1.6bb84p+4, 0x3d342ec5u},       {-0x1.7895cp+3, 0xbdae06f2u},
      {-0x1.116bp+5, 0xbcefb10fu},       {-0x1.7bd4p+1, 0xbeac8a8bu},
      {0x1.29620eb378p+25, 0x32dc602fu}, {0x1.29dd006dep+19, 0x35dc0539u},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(
                  amdgpu::raster::truncate_float(amdgpu::raster::area_reciprocal(test.area))),
              test.expected);
}

TEST(GraphicsRasterMathTest, AreaReciprocalMatchesPhysicalDepthGrid) {
  // Raw gl_FragCoord.z digest from 32768 triangles captured on RDNA3/4.
  // The numerator is exactly one; the regular area grid spans every seed segment.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t index = 0; index < 32768; ++index) {
    const double area = (2048 + index / 16.0) * 8;
    const amdgpu::raster::DepthPlane plane{
        amdgpu::raster::depth_gradient(1, amdgpu::raster::area_reciprocal(area)), 0, 0, 0, 0};
    for (int x = 0; x < 128; ++x)
      digest = (digest ^ std::bit_cast<uint32_t>(plane.at(x, 0))) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0x7854ef00633a3c21ull);
}

TEST(GraphicsRasterMathTest, AreaReciprocalTruncatesWideInputsBeforeSeed) {
  // Raw RDNA3/4 depth values from large triangles with numerator exactly one.
  // The last four areas become seed boundaries after input truncation, so
  // discarded input bits must not contribute to the seed's sticky bit.
  struct Case {
    double area;
    int x;
    uint32_t expected;
  };
  const Case cases[] = {
      {0x1.3a3e5945e8ep+29, 0, 0x30508d37u},   {0x1.c6caa591adp+26, 104, 0x356b4a4au},
      {0x1.085e7be51d8p+28, 48, 0x343bdbccu},  {0x1.71b5e6b5afep+28, 63, 0x342fe0cdu},
      {0x1.bbf68001aap+27, 52, 0x34722eb2u},   {0x1.5eca8000e64p+29, 0, 0x303ad2d0u},
      {0x1.065e00017f38p+29, 65, 0x33ffa448u}, {0x1.17078000d15p+29, 123, 0x34629d3cu},
  };
  for (const auto &test : cases) {
    const amdgpu::raster::DepthPlane plane{
        amdgpu::raster::depth_gradient(1, amdgpu::raster::area_reciprocal(test.area)), 0, 0, 0, 0};
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at(test.x, 0)), test.expected);
  }
}

TEST(GraphicsRasterMathTest, DepthGradientMatchesPhysicalSignedAndStickyInputs) {
  // Raw positive/negative depth captures distinguish seed sticky bits, the
  // gradient product width, and truncation toward zero from rounding downward.
  struct Case {
    double area;
    float numerator;
    double base;
    int x;
    uint32_t expected;
  };
  const Case cases[] = {
      {0x1.1b09a00000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 57, 0x3b500765u},
      {0x1.1c03000000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 37, 0x3b0734a7u},
      {0x1.34a3600000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 20, 0x3a8807a3u},
      {0x1.548fa00000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 81, 0x3b750df0u},
      {0x1.680ba00000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 3, 0x391f44cfu},
      {0x1.a04e200000000p+14, 0x1.0000000000000p+0f, 0x0.0p+0, 17, 0x3a2c2e6eu},
      {0x1.e3bd818000000p+16, 0x1.c93f360000000p+0f, 0x0.0p+0, 21, 0x39a29477u},
      {0x1.29ef6ec000000p+17, 0x1.5645080000000p-3f, 0x0.0p+0, 34, 0x381e88feu},
      {0x1.15f0bcc000000p+17, 0x1.3efa920000000p+5f, 0x0.0p+0, 65, 0x3c9657a6u},
      {0x1.6ecbb6b000000p+17, 0x1.2c06320000000p+5f, 0x0.0p+0, 39, 0x3c013cd6u},
      {0x1.3bb2c28000000p+16, 0x1.c225160000000p+3f, 0x0.0p+0, 66, 0x3c3da409u},
      {0x1.f684445000000p+17, 0x1.2053500000000p+5f, 0x0.0p+0, 105, 0x3c7220c1u},
      {0x1.aa767b8000000p+18, -0x1.0000020000000p+0f, 0x1.0000000000000p-1, 21, 0x3efff98cu},
      {0x1.9192488000000p+16, -0x1.00000e0000000p+0f, 0x1.0000000000000p-1, 110, 0x3eff731du},
      {0x1.452dcb0000000p+14, -0x1.0000080000000p+0f, 0x1.0000000000000p-1, 120, 0x3efd0915u},
      {0x1.800b140000000p+15, -0x1.ffffe20000000p-1f, 0x1.0000000000000p-1, 47, 0x3eff8159u},
      {0x1.e1d91a8000000p+14, -0x1.0000040000000p+0f, 0x1.0000000000000p-1, 38, 0x3eff5c5du},
      {0x1.f9d8780000000p+13, -0x1.fffffc0000000p-1f, 0x1.0000000000000p-1, 78, 0x3efd845cu},
  };
  for (const auto &test : cases) {
    const double inverse = amdgpu::raster::area_reciprocal(test.area);
    const amdgpu::raster::DepthPlane plane{amdgpu::raster::depth_gradient(test.numerator, inverse),
                                           0, test.base, 0, 0};
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at(test.x, 0)), test.expected);
  }
}

TEST(GraphicsRasterMathTest, NegativeDepthSlopesMatchPhysicalPacking) {
  // Raw RDNA3/4 depth captures bracket 27-bit midpoint boundaries and translate
  // two-slope triangles across tile positions. Keep the original setup slopes
  // for the center; only negative local slopes round, with ties away from zero.
  struct Case {
    amdgpu::raster::DepthPlane plane;
    int x, y;
    uint32_t expected;
  };
  const Case cases[] = {
      {{-0x1.0708105e00000p-5, 0x0.0p+0, 0x1.fea6a80000000p-4, 0x0.0p+0, 0x0.0p+0},
       0,
       0,
       0x3dde7251u},
      {{-0x1.c03c36de00000p-5, 0x0.0p+0, 0x1.ac19840000000p-3, 0x0.0p+0, 0x0.0p+0},
       3,
       0,
       0x3c8f934fu},
      {{-0x1.088ea01e00000p-4, 0x0.0p+0, 0x1.dc68d80000000p-3, 0x0.0p+0, 0x0.0p+0},
       0,
       0,
       0x3e4d2297u},
      {{-0x1.bf5a09a000000p-11, 0x0.0p+0, 0x1.ea9a000000000p-9, 0x0.0p+0, 0x0.0p+0},
       2,
       0,
       0x3ad301bau},
      {{-0x1.05284ee000000p-5, 0x0.0p+0, 0x1.64cbd20000000p-3, 0x0.0p+0, 0x0.0p+0},
       3,
       0,
       0x3d80488du},
      {{-0x1.35ead22000000p-2, 0x0.0p+0, 0x1.3bf86a0000000p-1, 0x0.0p+0, 0x0.0p+0},
       1,
       0,
       0x3e271099u},
      {{-0x1.029a972200000p-8, 0x0.0p+0, 0x1.b0dbc20000000p-7, 0x0.0p+0, 0x0.0p+0},
       0,
       0,
       0x3c381a8eu},
      {{-0x1.be34b06200000p-6, 0x0.0p+0, 0x1.7f05bc0000000p-4, 0x0.0p+0, 0x0.0p+0},
       0,
       0,
       0x3da39f93u},
      {{-0x1.276a5a6200000p-9, 0x0.0p+0, 0x1.c5cc200000000p-7, 0x0.0p+0, 0x0.0p+0},
       1,
       0,
       0x3c2b821fu},
      {{-0x1.03b33c2000000p-3, -0x1.3510a7e400000p-3, 0x1.59b10c0000000p-2, 0x1.2b00000000000p+3,
        0x1.f600000000000p+0},
       7,
       3,
       0x3eada23au},
      // Tiny negative slopes shift out before rounding. The remaining witnesses
      // distinguish the raw slope, strict cutoff, and per-tile exponent.
      {{-0x1.66cfe68ep-33, -0x1.7e9abc78p-21, 0x1.ffe4dcp-1, 0x1.c665p+8, 0x1.1654p+7},
       423,
       0,
       0x3f7ff8e8u},
      {{-0x1.fffffffep-39, 0x1.85fffffep-32, 0x1.82bed4p-7, -0x1.36d4p+6, -0x1.c8p+0},
       14,
       4,
       0x3c415f6cu},
      {{-0x1p-32, 0x1.0fap-26, 0x1.0040b6p-1, -0x1.2e84p+6, 0x1.70b8p+5}, 41, 51, 0x3f00205cu},
      {{-0x1.68p-38, -0x1.1a2c6p-24, 0x1.ffde04p-7, -0x1.16d3p+8, 0x1.8588p+6}, 55, 1, 0x3c8004b6u},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(test.plane.at(test.x, test.y)), test.expected);
}

TEST(GraphicsRasterMathTest, DepthBiasMatchesPhysicalRdna3AndRdna4) {
  // Raw gl_FragCoord.z witnesses from polygon-offset triangles captured on both
  // cards. Together they separate FP32 gradient and product truncation, the
  // aligned sum and its 24-binade drop, the zero-depth exponent, every denormal
  // flush, and both clamp signs from nearby alternatives.
  struct Case {
    float numerator_x, numerator_y;
    double inverse_area, base, origin_x, origin_y;
    float largest_depth;
    bool enabled;
    uint32_t scale, offset, clamp, format;
    int x, y;
    uint32_t expected;
  };
  const Case cases[] = {
      // Crysis shadow pixels (898,432) and (873,278).
      {-0x1.9f48fp-12f, 0x1.cd868p-16f, -0x1.a298527cp-6, 0x1.fbac2p-2, 0x1.bec68p+9, 0x1.af2dp+8,
       0x1.fbbd4ap-2f, true, 0x40cccccd, 0x00000000, 0x3ad1b717, 0x1e9, 898, 432, 0x3efddd08},
      {-0x1.194p-21f, -0x1.3acccp-13f, -0x1.39ce91fcp-2, 0x1.fa6acep-2, 0x1.b3cc8p+9, 0x1.16b9p+8,
       0x1.fa6ef2p-2f, true, 0x40cccccd, 0x00000000, 0x3ad1b717, 0x1e9, 873, 278, 0x3efd3682},
      // An operand 24 binades below the other contributes nothing.
      {-0x1.f6d93cp+4f, -0x1.2d25f2p+2f, 0x1.3d91c326p-17, 0x1.c60f1cp-2, -0x1.26cp+6, -0x1.2b1p+6,
       0x1.c60f1cp-2f, true, 0xb644ec64, 0x3a062908, 0x00000000, 0x0, 2, 4, 0x3ed5e1a8},
      // A negative-zero clamp is disabled.
      {0x1.696fccp+0f, 0x1.30a2e4p+2f, -0x1.12085a3ep-13, 0x1.fed172p-2, -0x1.aa64p+6, 0x1.630cp+7,
       0x1.fed172p-2f, true, 0xc2c52c51, 0x30b5ec82, 0x80000000, 0x0, 40, 167, 0x3ef2be94},
      // Shifted-out bits round toward negative infinity.
      {0x1.36a998p+2f, -0x1.bbb8b8p+3f, 0x1.417d073cp-17, 0x1.44a54p-2, -0x1.3a2cp+6, 0x1.017dp+8,
       0x1.804fd6p-2f, true, 0xc2bdfcdb, 0xae9122e3, 0x00000000, 0x0, 29, 17, 0x3eb4d0e2},
      // A positive clamp limits the sum.
      {-0x1.1d2c1ep-1f, 0x1.b12fecp-5f, -0x1.05c820b8p-6, 0x1.ee0694p-2, 0x1.36c8p+6, 0x1.f11p+5,
       0x1.1431cp-1f, true, 0x40cccccd, 0x00000000, 0x3ad1b717, 0x1e9, 82, 57, 0x3f07dd3e},
      // A zero largest depth scales the constant by 2^-127.
      {0x0.0p+0f, 0x0.0p+0f, 0x1.a36e2ebp-18, 0x0.0p+0, -0x1.9p+6, -0x1.9p+6, 0x0.0p+0f, true,
       0x42000000, 0x5d800000, 0x00000000, 0x1e9, 0, 0, 0x12800000},
      // Denormal depths flush to zero.
      {0x0.0p+0f, 0x0.0p+0f, 0x1.a36e2ebp-18, 0x1.0p-130, -0x1.9p+6, -0x1.9p+6, 0x1.0p-130f, false,
       0x00000000, 0x00000000, 0x00000000, 0x1e9, 0, 0, 0x00000000},
      {0x0.0p+0f, 0x0.0p+0f, 0x1.a36e2ebp-18, 0x1.0p-126, -0x1.9p+6, -0x1.9p+6, 0x1.0p-126f, true,
       0x00000000, 0x80c00000, 0x00000000, 0x0, 0, 0, 0x80000000},
      // Gradients below 2^-122 flush per axis; 2^-122 itself survives.
      {0x1.4p-125f, 0x1.4p-126f, 0x1.47ae147ap-17, 0x1.0p-120, -0x1.0p+6, -0x1.0p+6, 0x1.0008p-120f,
       false, 0x00000000, 0x00000000, 0x00000000, 0x1e9, 0, 0, 0x03800000},
      {0x1.0p-106f, 0x0.0p+0f, 0x1.0p-16, 0x1.0p-126, -0x1.0p+6, -0x1.0p+6, 0x1.001p-114f, false,
       0x00000000, 0x00000000, 0x00000000, 0x1e9, 63, 0, 0x05ff2000},
      {0x1.fffep-107f, 0x0.0p+0f, 0x1.0p-16, 0x1.0p-126, -0x1.0p+6, -0x1.0p+6, 0x1.000fp-114f,
       false, 0x00000000, 0x00000000, 0x00000000, 0x1e9, 63, 0, 0x00800000},
      {0x1.0p-94f, -0x1.0p-108f, 0x1.0p-16, 0x1.0p-100, -0x1.0p+6, -0x1.0p+6, 0x1.4p-100f, false,
       0x00000000, 0x00000000, 0x00000000, 0x1e9, 0, 63, 0x0d881000},
      // A denormal sum, slope term, or constant term flushes to zero.
      {0x1.3ffd8p-103f, 0x1.3ffbp-104f, 0x1.47ae147ap-17, 0x1.0p-126, -0x1.0p+6, -0x1.0p+6,
       0x1.0p-111f, true, 0x41800000, 0x83ccab33, 0x00000000, 0x0, 0, 0, 0x071acf30},
      {0x1.3ffd8p-103f, 0x1.3ffbp-104f, 0x1.47ae147ap-17, 0x1.0p-126, -0x1.0p+6, -0x1.0p+6,
       0x1.0p-111f, true, 0x3c800000, 0x00800000, 0x00000000, 0x0, 0, 0, 0x071ad330},
      {0x1.3ffd8p-103f, 0x1.3ffbp-104f, 0x1.47ae147ap-17, 0x1.0p-126, -0x1.0p+6, -0x1.0p+6,
       0x1.0p-111f, true, 0x41000000, 0x00400000, 0x00000000, 0x0, 0, 0, 0x071b9bfb},
  };
  for (const auto &test : cases) {
    amdgpu::raster::DepthPlane plane{
        amdgpu::raster::depth_gradient(test.numerator_x, test.inverse_area),
        amdgpu::raster::depth_gradient(test.numerator_y, test.inverse_area), test.base,
        test.origin_x, test.origin_y};
    if (test.enabled)
      plane.base += amdgpu::raster::depth_bias(
          plane.dx, plane.dy, std::bit_cast<float>(test.scale) / 16.0f,
          amdgpu::raster::depth_bias_constant(std::bit_cast<float>(test.offset), test.format,
                                              test.largest_depth),
          std::bit_cast<float>(test.clamp));
    EXPECT_EQ(std::bit_cast<uint32_t>(plane.at(test.x, test.y)), test.expected);
  }
}

TEST(GraphicsRasterMathTest, ViewportDepthClampStoresNegativeZeroAsMinimum) {
  // A negative denormal biased depth reads back as -0 in gl_FragCoord.z on
  // RDNA3/4, while the clamped D32 attachment holds +0.
  const float flushed = -0.0f;
  EXPECT_EQ(std::bit_cast<uint32_t>(amdgpu::raster::clamp_viewport_depth(flushed, 0, 1)), 0u);
  EXPECT_EQ(amdgpu::raster::clamp_viewport_depth(flushed, 0.25f, 1), 0.25f);
  EXPECT_EQ(amdgpu::raster::clamp_viewport_depth(-0.5f, 0, 1), 0.0f);
  EXPECT_EQ(amdgpu::raster::clamp_viewport_depth(0.5f, 0, 1), 0.5f);
  EXPECT_EQ(amdgpu::raster::clamp_viewport_depth(1.5f, 0, 1), 1.0f);
}

TEST(GraphicsRasterMathTest, SubpixelQuantizationRoundsMidpointsToEven) {
  for (double value : {0.0, 17.0, -17.0}) {
    EXPECT_EQ(amdgpu::raster::round_subpixel(value + 0.5 / 256), value);
    EXPECT_EQ(amdgpu::raster::round_subpixel(value + 1.5 / 256), value + 2.0 / 256);
  }
}

TEST(GraphicsRasterMathTest, ViewportTruncationMatchesPhysicalRdna3AndRdna4) {
  // Controlled triangles distinguish truncating the multiply and the add from
  // a fused operation and from discarding aligned operand bits before adding.
  // Coordinates were recovered from raw hardware barycentric captures.
  struct Case {
    float position, w, scale, offset;
    int subpixel;
  };
  const Case cases[] = {
      {-0.8996930718421936f, 1, 210, 210, 5393},
      {-0.8993768692016602f, 1, 210, 210, 5410},
      {-0.8990606069564819f, 1, 210, 210, 5427},
      {-0.6084542870521545f, 1, 210, 210, 21049},
      {-0.6078218221664429f, 1, 210, 210, 21084},
      {-0.39026230573654175f, 1, 210, 210, 32780},
      {-0.3889974355697632f, 1, 210, 210, 32847},
      {0.0005859374068677425f, 1, 210, 210, 53791},
      {0.0012183779617771506f, 1, 210, 210, 53825},
      {0x1.719728p+3f, 0x1.2cded6p+4f, 210, 210, 86779},
      {0x1.4410a2p+4f, 0x1.4367dcp+4f, 160, 160, 82003},
      {0x1.ff166cp-1f, 0x1.0a0878p+2f, 64, 64, 20318},
  };
  for (const auto &test : cases)
    EXPECT_EQ(amdgpu::raster::viewport_coordinate(test.position, test.w, test.scale, test.offset),
              test.subpixel / 256.0);
}

TEST(GraphicsRasterMathTest, ViewportScalePrecedesPerspectiveDivision) {
  // Exact vertex inputs and subpixel coordinates recovered independently on
  // physical RDNA3 and RDNA4. These include the conditional-rendering triangle
  // and the buffer-device-address edge, plus a sweep over positive W values.
  struct Case {
    uint32_t position, w;
    double expected;
  };
  const Case cases[] = {
      {0xc0148f6e, 0x410f9103, 155.67578125}, {0x408c7bd1, 0x40a00000, 394.3828125},
      {0xbea6cd92, 0x3f3e6086, 118.00390625}, {0xbe9e11f8, 0x3f295d42, 112.00390625},
      {0xc11cde0d, 0x4129ce7e, 16.00390625},  {0x401d634c, 0x409be672, 316.0},
      {0xbeebb3df, 0x3f11952c, 40.00390625},  {0x40289793, 0x405d463d, 370.00390625},
      {0xc0cba720, 0x41686f4e, 118.00390625}, {0xc042be5d, 0x404c7b98, 10.00390625},
      {0xc0b5cd8b, 0x41378df1, 106.00390625}, {0xbff25182, 0x400756bc, 22.00390625},
      {0xc07311dd, 0x409103cc, 34.00390625},  {0xbf60af9f, 0x3fb01000, 76.00390625},
  };
  for (const auto &test : cases)
    EXPECT_EQ(amdgpu::raster::viewport_coordinate(std::bit_cast<float>(test.position),
                                                  std::bit_cast<float>(test.w), 210, 210),
              test.expected);
}

TEST(GraphicsRasterMathTest, ViewportDepthMatchesPhysicalRdna3AndRdna4) {
  // Raw gl_FragCoord.z captures from triangles with constant Z and W isolate
  // viewport arithmetic from depth interpolation. Include reversed and narrow
  // depth ranges, which distinguish scaling before division from scaling NDC.
  struct Case {
    float z, w, scale, offset;
    uint32_t expected;
  };
  const Case cases[] = {
      {0x1.18caaep+24f, 0x1.504b1ap+26f, 1, 0, 0x3e55bffa},
      {0x1.45e0d2p-5f, 0x1.8489a6p-5f, 1, 0, 0x3f56b701},
      {0x1.754f2p-7f, 0x1.26cd38p-6f, 1, 0, 0x3f221650},
      {0x1.18caaep+24f, 0x1.504b1ap+26f, .75f, .125f, 0x3e9027fd},
      {0x1.45e0d2p-5f, 0x1.8489a6p-5f, .75f, .125f, 0x3f410940},
      {0x1.36f0f2p+3f, 0x1.45d762p+6f, .75f, .125f, 0x3e5b9c2e},
      {0x1.45e0d2p-5f, 0x1.8489a6p-5f, -.75f, 1, 0x3ebded80},
      {0x1.e36a64p+12f, 0x1.a9922cp+13f, -.75f, 1, 0x3f12f392},
      {0x1.12b2aap-24f, 0x1.4e3484p-23f, -.75f, 1, 0x3f3117e7},
      {0x1.18caaep+24f, 0x1.504b1ap+26f, 0x1.47aep-8f, .31f, 0x3e9f411e},
      {0x1.e36a64p+12f, 0x1.a9922cp+13f, 0x1.47aep-8f, .31f, 0x3ea02c89},
      {0x1.62a5fcp-6f, 0x1.6553e8p-5f, 0x1.47aep-8f, .31f, 0x3e9ffd8a},
  };
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(
                  amdgpu::raster::viewport_transform(test.z, test.w, test.scale, test.offset)),
              test.expected);
}

TEST(GraphicsRasterMathTest, ReciprocalMatchesPhysicalRdna3AndRdna4) {
  // Raw pull-model reciprocal-W captures cover sticky-bit boundaries, seed
  // segment boundaries, values differing from IEEE division, and exponents.
  constexpr std::array<std::array<uint32_t, 2>, 24> cases{{
      {0x3f800000u, 0x3f800000u}, {0x3f800001u, 0x3f7ffffeu}, {0x3f8003ffu, 0x3f7ff802u},
      {0x3f800400u, 0x3f7ff800u}, {0x3f800401u, 0x3f7ff7feu}, {0x3f81ffffu, 0x3f7c0fc3u},
      {0x3f820000u, 0x3f7c0fc1u}, {0x3f820001u, 0x3f7c0fbfu}, {0x3fb55555u, 0x3f34b4b5u},
      {0x3fc00000u, 0x3f2aaaabu}, {0x3fc00001u, 0x3f2aaaaau}, {0x3fffffffu, 0x3f000001u},
      {0x3f800556u, 0x3f7ff555u}, {0x3f8d547cu, 0x3f67dac1u}, {0x3f9bd297u, 0x3f524a58u},
      {0x3fabe70du, 0x3f3e9ea1u}, {0x3fbd7fffu, 0x3f2ceb11u}, {0x3fd1446du, 0x3f1c959du},
      {0x3fe7be07u, 0x3f0d6601u}, {0x3fffeaa9u, 0x3f000aadu}, {0x02000000u, 0x7d000000u},
      {0x30000000u, 0x4f000000u}, {0x60000000u, 0x1f000000u}, {0x7e000000u, 0x01000000u},
  }};
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(amdgpu::raster::reciprocal(std::bit_cast<float>(test[0]))),
              test[1])
        << std::hex << test[0];
}

TEST(GraphicsRasterMathTest, ReciprocalCompleteNormalizedHardwareDigest) {
  // FNV-style hash of raw reciprocal-W words captured independently on gfx1100 and gfx1201.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t mantissa = 0; mantissa < (1u << 23); ++mantissa) {
    const float input = std::bit_cast<float>(0x3f800000u | mantissa);
    digest =
        (digest ^ std::bit_cast<uint32_t>(amdgpu::raster::reciprocal(input))) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0xdb90a52d4704a833ull);
}

TEST(GraphicsRasterMathTest, PerspectiveProductsMatchPhysicalRdna3AndRdna4) {
  // Raw barycentric captures with power-of-two plane gradients isolate the
  // interpolation multiplier from triangle setup and parameter interpolation.
  constexpr std::array<std::array<uint32_t, 3>, 4> cases{{
      {0x3d900000, 0x3f8b7034, 0x3d9cde3b},
      {0x3e580000, 0x3fa4a9cf, 0x3e8aef46},
      {0x3e600000, 0x3fa655c4, 0x3e918b0b},
      {0x3e400000, 0x3fa237c3, 0x3e7353a5},
  }};
  for (const auto &test : cases)
    EXPECT_EQ(std::bit_cast<uint32_t>(amdgpu::raster::multiply_perspective(
                  std::bit_cast<float>(test[0]), std::bit_cast<float>(test[1]))),
              test[2]);
}

TEST_P(GraphicsExportTest, ParameterLoadUsesQuadMaskAndPrimitiveOffsets) {
  // Quad two starts a second primitive; attribute one follows both records of attr0.
  wave_->set_m0(128 | (1u << 17));
  wave_->set_lds_base(1024);
  wave_->set_exec((1u << 1) | (1u << 9));
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
    wave_->debug_write_vgpr(3, lane, 0xdeadbeef);
  for (uint32_t primitive = 0; primitive < 2; ++primitive)
    for (uint32_t coefficient = 0; coefficient < 3; ++coefficient)
      lds_.write32(1024 + 128 + 96 + primitive * 48 + 24 + coefficient * 4,
                   100 + primitive * 10 + coefficient);
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    run(rdna4::build_vdsdir(0, {.vdst = 3, .attr_chan = 2, .attr = 1}));
  else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
    run(rdna3_5::build_ldsdir(0, {.vdst = 3, .attr_chan = 2, .attr = 1}));
  else
    run(rdna3::build_ldsdir(0, {.vdst = 3, .attr_chan = 2, .attr = 1}));
  EXPECT_FALSE(wave_->instruction_execution_failed());
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
    uint32_t expected = 0xdeadbeef;
    if (lane < 3)
      expected = 100 + lane;
    else if (lane >= 8 && lane < 11)
      expected = 110 + lane - 8;
    if (lane % 4 != 3) { // The unused fourth coefficient is unspecified.
      EXPECT_EQ(wave_->debug_read_vgpr(3, lane), expected) << lane;
    }
  }
}

TEST_P(GraphicsExportTest, FragmentInitializationPreservesInputPackingAndPadding) {
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  struct Restore {
    std::fenv_t &saved;
    ~Restore() { std::fesetenv(&saved); }
  } restore{saved};
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  constexpr uint32_t input_bits[] = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15};
  constexpr uint32_t first[] = {0, 2, 4, 6, 9, 11, 13, 15, 16, 17, 18, 19, 20};
  constexpr uint32_t count[] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1};
  constexpr uint32_t barycentric_mask = (1u << 7) - 1;
  constexpr uint32_t remaining_mask = (1u << (std::size(input_bits) - 7)) - 1;
  for (uint32_t wave_size : {32u, 64u}) {
    wave_->halt();
    wave_ = cu_->dispatch_wf(0, 0, 106, 32, wave_size);
    ASSERT_NE(wave_, nullptr);
    wave_->set_lds(&lds_);
    wave_->set_lds_base(128);
    wave_->set_wg_coord(0, 0, 0);
    amdgpu::GraphicsDraw draw(batch_state(wave_size), GetParam(), 3);
    amdgpu::GraphicsDrawTestAccess::seed_fragment_inputs(draw, wave_size);
    for (uint32_t selection = 0; selection < (1u << std::size(input_bits)); ++selection) {
      const uint32_t barycentric = selection & barycentric_mask;
      const uint32_t remaining = selection >> 7;
      // Cover every barycentric combination with the remaining inputs absent
      // or complete. Exercise every remaining-input combination after 0, 2 and
      // 15 barycentric VGPRs. These 442 masks cover every input's possible VGPR
      // offset and every output length without crossing all 8192 combinations.
      if (remaining != 0 && remaining != remaining_mask && barycentric != 0 && barycentric != 1 &&
          barycentric != barycentric_mask)
        continue;
      SCOPED_TRACE(testing::Message() << wave_size << ',' << selection);
      uint32_t mask = 0;
      for (uint32_t input = 0; input < std::size(input_bits); ++input)
        mask |= ((selection >> input) & 1) << input_bits[input];
      amdgpu::GraphicsDrawTestAccess::set_fragment_inputs(draw, mask);
      for (uint32_t reg = 0; reg < 22; ++reg)
        for (uint32_t lane = 0; lane < wave_size; ++lane)
          wave_->debug_write_vgpr(reg, lane, 0xdeadbeef);
      draw.initialize(*wave_, 0, 0);
      EXPECT_EQ(wave_->exec(), 0xa5u);
      EXPECT_EQ(wave_->debug_read_sgpr(0), 0x76543210u);
      EXPECT_EQ(wave_->debug_read_sgpr(1), 0xfedcba98u);
      EXPECT_EQ(wave_->debug_read_sgpr(2), 0u);
      if (selection == 0) {
        wave_->set_m0(wave_->debug_read_sgpr(2));
        amdgpu::execute_graphics_parameter_load(*wave_, 31, 0, 0);
        EXPECT_EQ(wave_->debug_read_vgpr(31, 0), 0x01234567u);
        EXPECT_EQ(wave_->debug_read_vgpr(31, 1), 0x89abcdefu);
        EXPECT_EQ(wave_->debug_read_vgpr(31, 2), 0x7f801234u);
      }
      EXPECT_EQ(lds_.read32(128), 0x01234567u);
      EXPECT_EQ(lds_.read32(132), 0x89abcdefu);
      EXPECT_EQ(lds_.read32(136), 0x7f801234u);
      for (uint32_t lane = 0; lane < wave_size; ++lane) {
        const bool populated = lane < 8; // Includes helper lanes with EXEC clear.
        const uint32_t i = populated ? 0x3f000001u + lane : 0;
        const uint32_t j = populated ? 0x80000000u | lane : 0;
        const uint32_t linear_i = populated ? 1u + lane : 0;
        const uint32_t linear_j = populated ? 0xbf800000u + lane : 0;
        const int32_t x = populated ? int32_t(lane) - 3 : 0;
        const int32_t y = populated ? 2 - int32_t(lane) : 0;
        const std::array<uint32_t, 21> canonical{i,
                                                 j,
                                                 i,
                                                 j,
                                                 i,
                                                 j,
                                                 populated ? 0x7f800001u + lane : 0,
                                                 populated ? 0x80000000u : 0,
                                                 populated ? 0x3f800000u : 0,
                                                 linear_i,
                                                 linear_j,
                                                 linear_i,
                                                 linear_j,
                                                 linear_i,
                                                 linear_j,
                                                 std::bit_cast<uint32_t>(float(x) + 0.5f),
                                                 std::bit_cast<uint32_t>(float(y) + 0.5f),
                                                 populated ? 0x7fc00000u + lane : 0,
                                                 populated ? 0x3f800000u : 0x7f800000u,
                                                 0x12340000u,
                                                 (uint32_t(x) & 0xffff) | (uint32_t(y) << 16)};
        std::array<uint32_t, 22> expected, actual;
        expected.fill(0xdeadbeef);
        uint32_t reg = 0;
        for (uint32_t input = 0; input < std::size(input_bits); ++input)
          if (selection & (1u << input))
            for (uint32_t component = 0; component < count[input]; ++component)
              expected[reg++] = canonical[first[input] + component];
        for (uint32_t r = 0; r < actual.size(); ++r)
          actual[r] = wave_->debug_read_vgpr(r, lane);
        EXPECT_EQ(actual, expected) << lane;
      }
    }
  }
}

TEST_P(GraphicsExportTest, UnusedLinearPlanesKeepConsumedInputsAndHostState) {
#if defined(__GLIBC__) && defined(__x86_64__)
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  amdgpu::CpuDispatchPool pool(4);
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (uint32_t size : {4u, 65u}) {
    for (uint32_t wave_size : {32u, 64u}) {
      // Test every LINEAR address bit independently, including ADDR without
      // ENA, and the caller's observer refusal. The large case uses the pool.
      for (uint32_t mode = 0; mode < 5; ++mode) {
        SCOPED_TRACE(size);
        SCOPED_TRACE(wave_size);
        SCOPED_TRACE(mode);
        ASSERT_EQ(std::fesetenv(FE_DFL_ENV), 0);
        auto state = batch_state(wave_size);
        auto &ctx = state.context_registers;
        ctx[gfx12 ? 0x190 : 0x1b6] = (wave_size == 32 ? 1u << 15 : 0) | 1u;
        if (gfx12)
          state.sh_registers[0x31] = 1u << 11;
        ctx[gfx12 ? 0x199 : 0x191] = 0;
        for (uint32_t vertex = 0; vertex < 3; ++vertex)
          for (uint32_t component = 0; component < 4; ++component)
            memory_.write32(0x10000 + vertex * 16 + component * 4,
                            0x3f000001u + vertex * 4 + component);
        ctx[gfx12 ? 0x198 : 0x1b4] = 0x302 | (mode > 0 && mode < 4 ? 1u << (mode + 3) : 0);
        ctx[gfx12 ? 0x197 : 0x1b3] = 0; // ADDR alone defines initialized registers.
        ctx[gfx12 ? 0x31e : 0x3b0] = (size - 1) | ((size - 1) << (gfx12 ? 16 : 14));
        ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] =
            std::bit_cast<uint32_t>(float(size) / 2);
        ctx[0x91] = (size - gfx12) | ((size - gfx12) << 16);
        ctx[0x30f] = 1; // Keep uncovered helper lanes and a padded final wave.
        auto original = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        export_rectangle_vertices(*original);
        std::fenv_t before;
        ASSERT_EQ(std::fegetenv(&before), 0);
        before.__status_word &= ~0x3fu;
        before.__mxcsr = (before.__mxcsr & ~0x3fu) | 0x20u;
        std::array<std::vector<uint32_t>, 2> fragments, initialized;
        std::array<std::fenv_t, 2> after, initialized_environment;
        std::array<int, 2> errors;
        for (uint32_t optimized = 0; optimized < 2; ++optimized) {
          auto draw = std::make_shared<amdgpu::GraphicsDraw>(*original);
          amdgpu::GraphicsDrawTestAccess::poison_retired_fragment_storage(*draw, 1024);
          ASSERT_EQ(std::fesetenv(&before), 0);
          errno = E2BIG;
          const auto dispatch =
              draw->advance(*access_, size == 65 ? &pool : nullptr, 4, optimized && mode != 4);
          errors[optimized] = errno;
          ASSERT_EQ(std::fegetenv(&after[optimized]), 0);
          ASSERT_TRUE(dispatch);
          EXPECT_EQ(dispatch->total_wgs > 32, size == 65);
          fragments[optimized] = amdgpu::GraphicsDrawTestAccess::consumed_fragment_bits(*draw);
          const bool omitted = optimized && mode == 0 && amdgpu::raster::supports_sse41_planes();
          EXPECT_EQ(amdgpu::GraphicsDrawTestAccess::linear_values_are_zero(*draw), omitted);
          if (mode != 0 || !amdgpu::raster::supports_sse41_planes())
            fragments[optimized] = amdgpu::GraphicsDrawTestAccess::fragment_bits(*draw);
          for (uint32_t group = 0; group < dispatch->total_wgs; ++group) {
            batch_wave(wave_size, group);
            for (uint32_t reg = 0; reg < 16; ++reg)
              for (uint32_t lane = 0; lane < wave_size; ++lane)
                wave_->debug_write_vgpr(reg, lane, 0xdead0000u + reg * 64 + lane);
            draw->initialize(*wave_, 0, 0);
            initialized[optimized].push_back(wave_->exec());
            initialized[optimized].push_back(wave_->exec() >> 32);
            for (uint32_t reg = 0; reg < 16; ++reg)
              for (uint32_t lane = 0; lane < wave_size; ++lane)
                initialized[optimized].push_back(wave_->debug_read_vgpr(reg, lane));
          }
          ASSERT_EQ(std::fegetenv(&initialized_environment[optimized]), 0);
        }
        EXPECT_EQ(fragments[0], fragments[1]);
        EXPECT_EQ(initialized[0], initialized[1]);
        EXPECT_EQ(errors[0], errors[1]);
        EXPECT_EQ(after[0].__control_word, after[1].__control_word);
        EXPECT_EQ(after[0].__status_word, after[1].__status_word);
        EXPECT_EQ(after[0].__mxcsr, after[1].__mxcsr);
        EXPECT_EQ(initialized_environment[0].__control_word,
                  initialized_environment[1].__control_word);
        EXPECT_EQ(initialized_environment[0].__status_word,
                  initialized_environment[1].__status_word);
        EXPECT_EQ(initialized_environment[0].__mxcsr, initialized_environment[1].__mxcsr);
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST_P(GraphicsExportTest, FragmentInitializationKeepsFallbackAndHostState) {
#if defined(__GLIBC__) && defined(__x86_64__)
  class Observer final : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("fragment_initialization") {}
    uint32_t writes = 0;
    void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *, uint32_t, uint64_t, uint8_t) override {
      ++writes;
    }
  };
  auto plugins = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Observer>();
  auto *observed = observer.get();
  plugins->add(std::move(observer));
  cu_->set_plugin_group(plugins);
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (uint32_t wave_size : {32u, 64u}) {
    amdgpu::GraphicsDraw draw(batch_state(wave_size), GetParam(), 3);
    amdgpu::GraphicsDrawTestAccess::seed_fragment_inputs(draw, wave_size);
    for (uint32_t mask : {0u, 0x7fu, 0xef7fu}) {
      amdgpu::GraphicsDrawTestAccess::set_fragment_inputs(draw, mask);
      for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        for (uint32_t flush : {0u, 0x40u, 0x8000u, 0x8040u}) {
          std::array<uint32_t, 22 * 64> expected{};
          for (uint32_t allocated : {32u, 1u}) {
            wave_->halt();
            wave_ = cu_->dispatch_wf(0, 0, 106, allocated, wave_size);
            ASSERT_NE(wave_, nullptr);
            wave_->set_lds(&lds_);
            wave_->set_wg_coord(0, 0, 0);
            ASSERT_EQ(std::fesetround(rounding), 0);
            std::fenv_t before;
            ASSERT_EQ(std::fegetenv(&before), 0);
            before.__mxcsr = (before.__mxcsr & ~0x8040u) | flush;
            ASSERT_EQ(std::fesetenv(&before), 0);
            ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
            ASSERT_EQ(std::feraiseexcept(FE_OVERFLOW), 0);
            ASSERT_EQ(std::fegetenv(&before), 0);
            const int before_flags = std::fetestexcept(FE_ALL_EXCEPT);
            errno = EDOM;
            draw.initialize(*wave_, 0, 0);
            const int error = errno;
            std::fenv_t after;
            ASSERT_EQ(std::fegetenv(&after), 0);
            EXPECT_EQ(error, EDOM);
            EXPECT_EQ(after.__control_word, before.__control_word);
            EXPECT_EQ(after.__mxcsr & ~0x3fu, before.__mxcsr & ~0x3fu);
            // Padding lanes evaluate 1/0 even when no position input is enabled.
            EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), before_flags | FE_DIVBYZERO);
            EXPECT_EQ(observed->writes, 0u);
            for (uint32_t reg = 0; reg < 22; ++reg)
              for (uint32_t lane = 0; lane < wave_size; ++lane) {
                const uint32_t bits = wave_->debug_read_vgpr(reg, lane);
                if (allocated == 32) {
                  expected[reg * 64 + lane] = bits;
                } else {
                  EXPECT_EQ(bits, expected[reg * 64 + lane]) << reg << ',' << lane;
                }
              }
          }
        }
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST(GraphicsFragmentInitializationTest, FirstTouchKeepsChunkAndArithmeticOrder) {
#if defined(__GLIBC__) && defined(__x86_64__)
  using Base = amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, rdna3::Isa>;
  class RecordingCu final : public Base {
  public:
    using Base::Base;
    using Base::raw_vgpr_data;
    struct Touch {
      uint32_t reg;
      size_t chunks;
      int flags;
    };
    std::array<Touch, 21> touches{};
    size_t count = 0;
    uint8_t *raw_vgpr_data(uint32_t reg) override {
      if (count < touches.size())
        touches[count] = {reg, vgpr_file().materialized_chunk_count(),
                          std::fetestexcept(FE_ALL_EXCEPT)};
      ++count;
      return Base::raw_vgpr_data(reg);
    }
  };
  amdgpu::GpuMemory memory("fragment_initialization");
  amdgpu::L2Cache cache("fragment_initialization_cache");
  cache.set_backing_memory(&memory);
  amdgpu::Lds lds(4);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA3;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  RecordingCu cu("recording", config, &memory, &cache);
  amdgpu::Pm4QueueState state;
  state.num_instances = 1;
  state.uconfig_registers[0x242] = 17;
  amdgpu::GraphicsDraw draw(state, config.arch, 3);
  amdgpu::GraphicsDrawTestAccess::seed_fragment_inputs(draw, 64, true);
  amdgpu::GraphicsDrawTestAccess::set_fragment_inputs(draw, 0xaf7f);
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  struct Restore {
    std::fenv_t &saved;
    ~Restore() { std::fesetenv(&saved); }
  } restore{saved};
  for (uint32_t reuse = 0; reuse < 2; ++reuse) {
    auto *wave = cu.dispatch_wf(0, 0, 106, 32, 64);
    ASSERT_NE(wave, nullptr);
    wave->set_lds(&lds);
    wave->set_wg_coord(0, 0, 0);
    EXPECT_EQ(cu.vgpr_file().materialized_chunk_count(), 0u);
    ASSERT_TRUE(wave->initialization_vgpr_lanes(32).empty());
    EXPECT_EQ(cu.vgpr_file().materialized_chunk_count(), 0u);
    cu.count = 0;
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    draw.initialize(*wave, 0, 0);
    ASSERT_EQ(cu.count, 21u);
    for (uint32_t reg = 0; reg < cu.touches.size(); ++reg) {
      EXPECT_EQ(cu.touches[reg].reg, reg);
      EXPECT_EQ(cu.touches[reg].chunks, reg == 0 ? 0u : reg <= 16 ? 1u : 2u);
      EXPECT_EQ(cu.touches[reg].flags, reg < 15 ? 0 : FE_DIVBYZERO);
    }
    EXPECT_EQ(cu.vgpr_file().materialized_chunk_count(), 2u);
    // Early pointers must still address their own lanes after the second chunk appears.
    EXPECT_EQ(wave->debug_read_vgpr(0, 7), 0x3f000008u);
    EXPECT_EQ(wave->debug_read_vgpr(18, 0), 0x7f800000u);
    EXPECT_EQ(wave->debug_read_vgpr(20, 7), 0xfffb0004u);
    wave->halt();
  }
#else
  GTEST_SKIP();
#endif
}

TEST(GraphicsFragmentInitializationTest, OutOfFileFallbackKeepsDroppedWrites) {
  using Base = amdgpu::IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, rdna3::Isa>;
  class Cu final : public Base {
  public:
    using Base::Base;
    using Base::raw_vgpr_data;
    bool refuse_storage = false;
    uint32_t storage_requests = 0;
    uint8_t *raw_vgpr_data(uint32_t reg) override {
      ++storage_requests;
      return refuse_storage ? nullptr : Base::raw_vgpr_data(reg);
    }
  };
  amdgpu::GpuMemory memory("fragment_bounds");
  amdgpu::L2Cache cache("fragment_bounds_cache");
  cache.set_backing_memory(&memory);
  amdgpu::Lds lds(4);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA3;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 16;
  Cu cu("bounded", config, &memory, &cache);
  auto *wave = cu.dispatch_wf(0, 0, 106, 16, 64);
  ASSERT_NE(wave, nullptr);
  wave->set_lds(&lds);
  wave->set_wg_coord(0, 0, 0);
  EXPECT_TRUE(wave->initialization_vgpr_lanes(16).empty());
  EXPECT_EQ(cu.vgpr_file().materialized_chunk_count(), 0u);
  amdgpu::Pm4QueueState state;
  state.num_instances = 1;
  state.uconfig_registers[0x242] = 17;
  amdgpu::GraphicsDraw draw(state, config.arch, 3);
  amdgpu::GraphicsDrawTestAccess::seed_fragment_inputs(draw, 64);
  amdgpu::GraphicsDrawTestAccess::set_fragment_inputs(draw, 0xaf7f);
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  struct Restore {
    std::fenv_t &saved;
    ~Restore() { std::fesetenv(&saved); }
  } restore{saved};
  draw.initialize(*wave, 0, 0);
  EXPECT_EQ(wave->debug_read_vgpr(0, 7), 0x3f000008u);
  EXPECT_EQ(wave->debug_read_vgpr(16, 0), 0u);
  EXPECT_EQ(wave->debug_read_vgpr(20, 7), 0u);
  EXPECT_EQ(cu.vgpr_file().materialized_chunk_count(), 1u);
  wave->halt();
  // A custom CU may decline raw storage while supporting its ordinary writes.
  wave = cu.dispatch_wf(0, 0, 106, 16, 64);
  ASSERT_NE(wave, nullptr);
  wave->set_lds(&lds);
  wave->set_wg_coord(0, 0, 0);
  cu.refuse_storage = true;
  EXPECT_TRUE(wave->initialization_vgpr_lanes(0).empty());
  EXPECT_EQ(cu.vgpr_file().materialized_chunk_count(), 0u);
  cu.storage_requests = 0;
  amdgpu::GraphicsDrawTestAccess::set_fragment_inputs(draw, 0);
  draw.initialize(*wave, 0, 0);
  EXPECT_EQ(cu.storage_requests, 0u);
  amdgpu::GraphicsDrawTestAccess::set_fragment_inputs(draw, 0xaf7f);
  draw.initialize(*wave, 0, 0);
  EXPECT_EQ(cu.storage_requests, 16u);
  EXPECT_EQ(wave->debug_read_vgpr(0, 7), 0x3f000008u);
  EXPECT_EQ(wave->debug_read_vgpr(15, 0), std::bit_cast<uint32_t>(-2.5f));
  EXPECT_EQ(wave->debug_read_vgpr(20, 7), 0u);
  wave->halt();
#ifdef NDEBUG
  // RegisterFile's oversized-request assertion is absent in release builds.
  // Retain the old dropped-write behavior even for this malformed dispatch.
  cu.refuse_storage = false;
  wave = cu.dispatch_wf(0, 0, 106, 32, 64);
  ASSERT_NE(wave, nullptr);
  wave->set_lds(&lds);
  wave->set_wg_coord(0, 0, 0);
  EXPECT_TRUE(wave->initialization_vgpr_lanes(16).empty());
  EXPECT_EQ(cu.vgpr_file().materialized_chunk_count(), 0u);
  draw.initialize(*wave, 0, 0);
  EXPECT_EQ(wave->debug_read_vgpr(0, 7), 0x3f000008u);
  EXPECT_EQ(wave->debug_read_vgpr(20, 7), 0u);
  EXPECT_EQ(cu.vgpr_file().materialized_chunk_count(), 1u);
  wave->halt();
#endif
}

TEST_P(GraphicsExportTest, RasterCoverageFragmentInputsAndUnsupportedStates) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  enum class Outcome { Draw, Empty, Reject };
  struct FragmentInputWitness {
    uint32_t xy;
    std::vector<uint32_t> values;
  };
  struct Case {
    const char *name;
    Outcome outcome = Outcome::Draw;
    uint32_t clip_control = 0;
    uint32_t shader_control = 0;
    uint32_t color_control = 0xcc0010;
    uint32_t target_mask = 15;
    uint32_t shader_mask = 15;
    uint32_t polygon_mode = 0;
    uint32_t samples = 0;
    uint32_t sample_coverage = 15;
    uint32_t inputs = 2;
    float z_scale = 0;
    std::array<float, 4> depth_bias{};
    bool depth_only = false;
    float depth = 0;
    bool first_vertex_depth_only = false;
    float extent = 1;
    bool full_scissor = false;
    bool triangle = false;
    uint32_t expected_coverage = 0;
    bool passthrough = false;
    bool flat = false;
    bool viewport_scissor = false;
    bool wide_window_scissor = false;
    std::optional<std::array<std::array<float, 4>, 3>> positions{};
    std::optional<FragmentInputWitness> fragment_inputs{};
    float guard = 16382.5f;
  };
  // Additional coverage masks were captured on physical gfx1100/gfx1201.
  const Case cases[] = {
      // This is a host-safety witness, not a physical interpolation oracle:
      // near clipping retains covered samples although the original snapped
      // screen vertices are collinear. Setup now uses the clipped vertices.
      // The interior [1,3) scissor avoids a snapped-polygon coverage boundary.
      {.name = "near clipping with collinear original snapped vertices",
       .clip_control = 1u << 19,
       .full_scissor = false,
       .triangle = true,
       .expected_coverage = 0x400,
       .positions =
           std::array<std::array<float, 4>, 3>{{{-1, -1, -1, 1},
                                                {0.5f + 3.0f / 2048, 0.5f + 5.0f / 2048, 1, 1},
                                                {1.5f + 5.0f / 2048, 1.5f + 3.0f / 2048, 1, 1}}}},
      {.name = "flat first provoking vertex", .inputs = 0xaf28, .flat = true},
      // Physical perspective I/J, noperspective I/J, and gl_FragCoord.z.
      // Depth clipping rebuilds setup planes with the rounded far distance.
      {.name = "near clipped inputs and depth",
       .clip_control = (1u << 19) | (1u << 24),
       .inputs = 0x8422,
       .z_scale = 1,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x230,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{-0x1.c8496ep+4f, 0x1.9df1p+5f, 0x1.6aa1a2p+4f, 0x1.d159b8p+5f},
                {-0x1.b8ac14p+12f, -0x1.8c3a2cp+17f, -0x1.85108ep+15f, 0x1.eb96f4p+19f},
                {-0x1.055f1ep+3f, -0x1.b24dc2p+2f, 0x1.d37004p+2f, 0x1.194c5p+3f}}},
       .fragment_inputs =
           FragmentInputWitness{0x10000,
                                {0x3d83993du, 0x3535bc12u, 0x3e9757a3u, 0x3d5cc8f4u, 0x3f2729f9u}}},
      {.name = "far clipped inputs and depth",
       .clip_control = (1u << 19) | (1u << 24),
       .inputs = 0x8422,
       .z_scale = 1,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x10,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{-0x1.2271d4p+1f, 0x1.aa99fp-3f, 0x1.9c1c6p+1f, 0x1.640a8ep+1f},
                {-0x1.5a4034p+13f, -0x1.58fd6ep+13f, 0x1.07b3a6p+13f, 0x1.f3da4ep+13f},
                {0x1.9e68fep-8f, -0x1.5132f2p-9f, 0x1.4ea3dcp-7f, 0x1.01573ep-7f}}},
       .fragment_inputs =
           FragmentInputWitness{0x10000,
                                {0x37a44469u, 0x3f58b17eu, 0x3ed6b7fcu, 0x3c11d308u, 0x3f65033bu}}},
      {.name = "near and far clipped inputs and depth",
       .clip_control = (1u << 19) | (1u << 24),
       .inputs = 0x8422,
       .z_scale = 1,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x40,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{0x1.cdddc6p-6f, -0x1.71adp-3f, 0x1.9bd90ep-2f, 0x1.79e09cp-2f},
                {0x1.487172p-9f, 0x1.b8495p-9f, -0x1.c1d228p-10f, 0x1.15b4fap-8f},
                {0x1.975676p+8f, 0x1.24d9c8p+8f, -0x1.3ff1cp+6f, 0x1.b6c514p+8f}}},
       .fragment_inputs =
           FragmentInputWitness{0x10002,
                                {0x38f74cd4u, 0x3f0ef79fu, 0x3e4c0428u, 0x3f4b26eeu, 0x3f535eddu}}},
      {.name = "far clipping interpolates near-plane residual",
       .clip_control = (1u << 19) | (1u << 24),
       .inputs = 0x8422,
       .z_scale = 1,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x660,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{0x1.95cca6p+19f, -0x1.09a082p+16f, 0x1.4a4e14p+20f, 0x1.2156ccp+20f},
                {0x1.5b6a88p+4f, 0x1.ffd61cp+7f, 0x1.fa36dcp+6f, 0x1.21266ep+8f},
                {-0x1.80d9a6p+1f, -0x1.608dc8p+1f, -0x1.6e003cp-3f, 0x1.ed7f2cp+1f}}},
       .fragment_inputs =
           FragmentInputWitness{0x10002,
                                {0x3986b3bau, 0x3f7feeb1u, 0x3bbf0f80u, 0x3e9ae40fu, 0x3f4731d3u}}},
      // Near-clipped W values differ by one ULP but share a setup reciprocal.
      // Choosing the smaller W rotates all four physical barycentric inputs.
      {.name = "clipped origin uses rounded reciprocal W",
       .clip_control = (1u << 19) | (1u << 24),
       .inputs = 0x8022,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xf0,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{0x1.dc9972p+0f, 0x1.152facp-4f, 0x1.9485c2p+1f, 0x1.9e79f6p+1f},
                {-0x1.80e7cap+2f, -0x1.466fc8p-3f, -0x1.a4759cp-3f, -0x1.a84da6p-4f},
                {0x1.a2bebcp-4f, -0x1.4047fp-4f, -0x1.723a6cp-3f, -0x1.448e44p-4f}}},
       .fragment_inputs =
           FragmentInputWitness{0x10000, {0x3f4f92d6u, 0x3ddc8196u, 0xbe6e6954u, 0x3fa1b7a4u}}},
      {.name = "guard clipping preserves tiny negative W coverage",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{-0.75f, -0.5f, 0, -0x1.8p-39f}, {0.75f, -0.5f, 0.25f, 1}, {0, 0.75f, 0.25f, 1}}}},
      {.name = "single-sample coverage input",
       .inputs = 0xc000,
       .fragment_inputs = FragmentInputWitness{0x10001, {1}}},
      {.name = "flat last provoking vertex",
       .polygon_mode = 1u << 19,
       .inputs = 0xaf28,
       .flat = true},
      {.name = "fragment reciprocal and perspective reconstruction",
       .clip_control = 1u << 19,
       .inputs = 0x880a,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0x1.7ffb74p-1f, -0x1.fff0f4p-2f,
                                                          0x1.80047ap-17f, 0x1.80047ap-15f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}},
       // Physical RDNA3/4 inputs: perspective I/J, pull I/W, J/W, 1/W, and W.
       .fragment_inputs =
           FragmentInputWitness{
               1, {0x3eb9eebfu, 0x3de83a50u, 0x3f4316d8u, 0x3e73aa10u, 0x40064dbau, 0x3ef3fc08u}}},
      {.name = "first clipped fan plane",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x6660,
       .positions = std::array<std::array<float, 4>, 3>{{{-0x1.a68p-4f, 0x1.0308p+1f, 0.5f, 1.0f},
                                                         {-0x1.03p-1f, -0x1.5f4p-1f, 0.5f, 1.0f},
                                                         {0x1.98ap-1f, -0x1.5f4p-1f, 0.5f, 1.0f}}},
       .fragment_inputs = FragmentInputWitness{0x20001, {0x3eb0d289u, 0x3f109cb1u, 0x3f800000u}},
       .guard = 1.0f},
      {.name = "second clipped fan plane",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x6660,
       .positions = std::array<std::array<float, 4>, 3>{{{-0x1.a68p-4f, 0x1.0308p+1f, 0.5f, 1.0f},
                                                         {-0x1.03p-1f, -0x1.5f4p-1f, 0.5f, 1.0f},
                                                         {0x1.98ap-1f, -0x1.5f4p-1f, 0.5f, 1.0f}}},
       .fragment_inputs = FragmentInputWitness{0x20002, {0x3ef221ecu, 0x3eb0d28du, 0x3f800000u}},
       .guard = 1.0f},
      {.name = "large guard-band plane 77",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xffff,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{0x1.4240f8p+12f, -0x1.5bd2d6p+12f, 0x1.9d4a52p-3f, 0x1.9d4a52p-2f},
                {-0x1.c59decp+3f, 0x1.0e3f8ep+7f, 0x1.01d442p-8f, 0x1.01d442p-7f},
                {-0x1.2ebf44p+10f, -0x1.5bea08p+10f, 0x1.9a73aap-5f, 0x1.9a73aap-4f}}},
       .fragment_inputs = FragmentInputWitness{0x0, {0x3f3f445du, 0x4262cbedu, 0x426fdb72u}}},
      {.name = "large guard-band plane 101",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xffff,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{0x1.a14c92p+11f, -0x1.6fd38cp+11f, 0x1.a0f93p-4f, 0x1.a0f93p-3f},
                {0x1.3f3308p+4f, 0x1.5d5316p+7f, 0x1.674696p-8f, 0x1.674696p-7f},
                {-0x1.582426p+9f, -0x1.33a608p+9f, 0x1.88e304p-6f, 0x1.88e304p-5f}}},
       .fragment_inputs = FragmentInputWitness{0x0, {0x40d2b26eu, 0x3f8db90eu, 0x424624b3u}}},
      {.name = "large guard-band plane 651",
       .clip_control = 1u << 19,
       .inputs = 0x8008,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xffff,
       .positions =
           std::array<std::array<float, 4>, 3>{
               {{-0x1.0e6bd6p+14f, -0x1.fc483p+13f, 0x1.222636p-1f, 0x1.222636p+0f},
                {0x1.f3d8eap+12f, -0x1.2deb56p+13f, 0x1.1a2406p-2f, 0x1.1a2406p-1f},
                {0x1.e1057p+6f, 0x1.595754p+9f, 0x1.4c13fep-6f, 0x1.4c13fep-5f}}},
       .fragment_inputs = FragmentInputWitness{0x0, {0x3ecb9e89u, 0x413d3646u, 0x4147d4b7u}}},
      {.name = "all vertices behind eye",
       .outcome = Outcome::Empty,
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x0,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.5f, -1.0f},
                                                         {0.75f, -0.5f, 0.5f, -1.0f},
                                                         {0.0f, 0.75f, 0.5f, -1.0f}}}},
      {.name = "top vertex behind eye",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xff0,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, -0.5f}}}},
      {.name = "left vertex behind eye",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, -0.5f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "two vertices behind eye",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x1,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.75f, -0.5f, 0.25f, -0.5f},
                                                         {0.0f, 0.75f, 0.25f, -0.5f}}}},
      {.name = "mixed W outside viewport",
       .outcome = Outcome::Empty,
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x0,
       .positions = std::array<std::array<float, 4>, 3>{{{-2.0f, -2.0f, 0.25f, 1.0f},
                                                         {2.0f, -2.0f, 0.25f, 1.0f},
                                                         {0.0f, -2.0f, 0.25f, -1.0f}}}},
      {.name = "near plane and eye clipping",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x7773,
       .positions = std::array<std::array<float, 4>, 3>{{{-1.25f, -0.5f, -0.25f, -0.5f},
                                                         {1.25f, -0.5f, 0.5f, 2.0f},
                                                         {0.0f, 1.75f, 0.125f, 0.5f}}}},
      {.name = "unequal W across eye plane",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xcc80,
       .positions = std::array<std::array<float, 4>, 3>{{{1.5f, 1.0f, 0.125f, -0.25f},
                                                         {-0.25f, 1.5f, 0.25f, 1.0f},
                                                         {0.75f, -1.0f, 0.5f, 2.0f}}}},
      {.name = "top vertex at zero W",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x7770,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 0.0f}}}},
      {.name = "left vertex at zero W",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 0.0f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "near-zero W",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1e-10f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "overflowing initial projection",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.25f, 1e-39f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "zero W and Z",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.0f, 0.0f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "subnormal W and Z",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x777,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 1e-39f, 1e-39f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "finite small W at zero Z",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x677,
       .positions = std::array<std::array<float, 4>, 3>{{{-0.75f, -0.5f, 0.0f, 1e-10f},
                                                         {0.75f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},
      {.name = "guard-band horizontal clipping",
       .clip_control = 1u << 19,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0xff0,
       .positions = std::array<std::array<float, 4>, 3>{{{-100000000.0f, -0.5f, 0.25f, 1.0f},
                                                         {100000000.0f, -0.5f, 0.25f, 1.0f},
                                                         {0.0f, 0.75f, 0.25f, 1.0f}}}},

      {.name = "integer coverage"},
      {.name = "viewport scissor", .expected_coverage = 0x440, .viewport_scissor = true},
      {.name = "unequal W and explicit vertex parameters", .inputs = 0xaf28, .passthrough = true},
      {.name = "position z and packed coordinates", .inputs = 0x8402},
      {.name = "unsupported line stipple input", .outcome = Outcome::Reject, .inputs = 0x80},
      {.name = "unsupported front-face input", .outcome = Outcome::Reject, .inputs = 0x1000},
      {.name = "fractional coverage", .extent = 0.625f, .full_scissor = true},
      {.name = "rasterizer discard", .outcome = Outcome::Empty, .clip_control = 1u << 22},
      {.name = "fragment discard enabled", .shader_control = 1u << 6},
      {.name = "early fragment tests", .outcome = Outcome::Reject, .shader_control = 1u << 12},
      {.name = "sample disabled", .outcome = Outcome::Empty, .sample_coverage = 0},
      {.name = "sample at even x and y", .sample_coverage = 1},
      {.name = "sample at odd x and even y", .sample_coverage = 2},
      {.name = "sample at even x and odd y", .sample_coverage = 4},
      {.name = "sample at odd x and y", .sample_coverage = 8},
      {.name = "fully clipped", .outcome = Outcome::Empty, .depth = 2},
      {.name = "partially clipped",
       .outcome = Outcome::Reject,
       .depth = 2,
       .first_vertex_depth_only = true},
      {.name = "triangle far clipping",
       .depth = 2,
       .first_vertex_depth_only = true,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x136},
      {.name = "triangle near clipping",
       .depth = -2,
       .first_vertex_depth_only = true,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x136},
      {.name = "triangle D3D near boundary",
       .outcome = Outcome::Empty,
       .clip_control = 1u << 19,
       .depth = -2,
       .first_vertex_depth_only = true,
       .triangle = true},
      {.name = "triangle far clipping disabled",
       .clip_control = 1u << 27,
       .depth = 2,
       .first_vertex_depth_only = true,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x137},
      {.name = "triangle near clipping disabled",
       .clip_control = (1u << 19) | (1u << 26),
       .depth = -2,
       .first_vertex_depth_only = true,
       .full_scissor = true,
       .triangle = true,
       .expected_coverage = 0x137},
      {.name = "pattern-dependent logic op", .outcome = Outcome::Reject, .color_control = 0x120010},
      {.name = "color disabled", .color_control = 0, .depth_only = true},
      {.name = "color mode disabled without depth", .color_control = 0},
      {.name = "all color writes masked", .target_mask = 0},
      {.name = "no fragment color export", .shader_mask = 0},
      {.name = "no attachment writes with viewport scissor",
       .target_mask = 0,
       .shader_mask = 0,
       .expected_coverage = 0x440,
       .viewport_scissor = true,
       .wide_window_scissor = true},
      {.name = "unsupported color mode", .outcome = Outcome::Reject, .color_control = 0xcc0020},
      {.name = "color degamma", .outcome = Outcome::Reject, .color_control = 0xcc0018},
      {.name = "line polygons",
       .outcome = Outcome::Reject,
       .polygon_mode = 8 | (1 << 5) | (1 << 8)},
      {.name = "point polygons", .outcome = Outcome::Reject, .polygon_mode = 8},
      {.name = "mixed polygon faces",
       .outcome = Outcome::Reject,
       .polygon_mode = 8 | (2 << 5) | (1 << 8)},
      {.name = "reserved polygon mode",
       .outcome = Outcome::Reject,
       .polygon_mode = 16 | (2 << 5) | (2 << 8)},
      {.name = "filled dual mode", .polygon_mode = 8 | (2 << 5) | (2 << 8)},
      {.name = "front depth bias", .polygon_mode = 1u << 11, .depth_bias = {1, 0, 0, 0}},
      {.name = "back depth bias", .polygon_mode = 1u << 12, .depth_bias = {0, 0, 1, 0}},
      {.name = "parallel depth bias",
       .outcome = Outcome::Reject,
       .polygon_mode = 1u << 13,
       .depth_bias = {0, 1, 0, 1}},
      {.name = "four samples", .outcome = Outcome::Reject, .samples = 2},
      {.name = "zero depth bias", .polygon_mode = 7u << 11, .depth_bias = {-0.0f, 0, 0, 0}},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    for (uint32_t y = 0; y < 4; ++y)
      for (uint32_t x = 0; x < 4; ++x) {
        const auto address = gfx12 ? amdgpu::gfx12_image_offset(x, y, 4, 4, 3)
                                   : amdgpu::gfx11_image_offset(x, y, 4, 4, 26);
        memory_.write32(0x100000 + *address, 0);
      }
    amdgpu::Pm4QueueState state;
    state.num_instances = 1;
    state.uconfig_registers[0x242] = test.triangle ? 4 : 0x11;
    state.context_registers[0x3b0] = 10;
    state.context_registers[0x31e] = (3 << 16) | 3;
    state.context_registers[0x31f] = 3 << 15;
    state.context_registers[0x318] = 0x1000;
    state.context_registers[0x214] = test.target_mask;
    state.context_registers[0x215] = test.shader_mask;
    state.context_registers[0x195] = 4;
    state.context_registers[0x198] = test.inputs;
    state.context_registers[0x113] = std::bit_cast<uint32_t>(test.z_scale);
    state.context_registers[0x2f9] = 0x2d;
    state.context_registers[0x205] = 0x43f;
    state.context_registers[0x10f] = state.context_registers[0x110] =
        state.context_registers[0x111] = state.context_registers[0x112] =
            std::bit_cast<uint32_t>(2.0f);
    state.context_registers[0x90] = 1 | (1 << 16);
    state.context_registers[0x91] = (3 - gfx12) | ((3 - gfx12) << 16);
    if (!gfx12) {
      state.context_registers[0x31c] = 10;
      state.context_registers[0x3b0] = 3 | (3 << 14);
      state.context_registers[0x3b8] = 26 << 14;
      state.context_registers[0x31e] = 0;
      state.context_registers[0x8e] = test.target_mask;
      state.context_registers[0x8f] = test.shader_mask;
      state.context_registers[0x1c5] = 4;
      state.context_registers[0x1b4] = test.inputs;
      state.context_registers[0x206] = 0x43f;
      state.context_registers[0x205] = 0;
    }
    if (test.full_scissor) {
      state.context_registers[0x90] = 0;
      state.context_registers[0x91] = (4 - gfx12) | ((4 - gfx12) << 16);
    }
    state.context_registers[0x204] = test.clip_control;
    state.context_registers[gfx12 ? 0x1b : 0x203] = test.shader_control;
    state.context_registers[gfx12 ? 0x216 : 0x202] = test.color_control;
    state.context_registers[gfx12 ? 0x207 : 0x205] = test.polygon_mode;
    state.context_registers[0x2f8] = test.samples;
    state.context_registers[0x30e] =
        (test.sample_coverage & 1) | ((test.sample_coverage & 2) << 15);
    state.context_registers[0x30f] =
        ((test.sample_coverage & 4) >> 2) | ((test.sample_coverage & 8) << 13);
    for (uint32_t i = 0; i < test.depth_bias.size(); ++i)
      state.context_registers[0x2e0 + i] = std::bit_cast<uint32_t>(test.depth_bias[i]);
    if (test.depth_only) {
      state.context_registers[gfx12 ? 0x1c : 0x200] = 6 | (7 << 4);
      state.context_registers[gfx12 ? 5 : 7] = (3 << 16) | 3;
      state.context_registers[gfx12 ? 6 : 0x10] = 3 | ((gfx12 ? 3 : 24) << 4);
      state.context_registers[gfx12 ? 8 : 0x12] = 0x2000;
      state.context_registers[gfx12 ? 10 : 0x14] = 0x2000;
    }
    const bool vertex_parameters = test.passthrough || test.flat;
    if (vertex_parameters) {
      // Leave the driver-owned descriptor table unmapped. Interpolation uses
      // SPI_ATTRIBUTE_RING_BASE, not the shader's user SGPRs.
      state.sh_registers[gfx12 ? 0x84 : 0x88] = 0xdead0000;
      state.uconfig_registers[0x446] = 0x12340;
      if (gfx12)
        state.sh_registers[0x31] = (1u << 11) | 1;
      else {
        state.context_registers[0x1b6] = 1;
        state.context_registers[0x1b1] = 2;
      }
      state.context_registers[gfx12 ? 0x199 : 0x191] = test.flat ? 0x401 : 0x421;
      state.context_registers[gfx12 ? 0x197 : 0x1b3] = test.inputs;
      // Attribute one follows 32 sixteen-byte records of attribute zero.
      for (uint32_t k = 0; k < 3; ++k)
        for (uint32_t c = 0; c < 4; ++c)
          memory_.write32(0x123400200ull + k * 16 + c * 4,
                          test.flat ? 0x7fc00000u + k * 4 + c
                                    : std::bit_cast<uint32_t>(float(10 + k * 4 + c)));
    }
    if (test.viewport_scissor) {
      state.context_registers[0x292] = 2;
      state.context_registers[0x94] = 2 | (1 << 16);
      state.context_registers[0x95] = (3 - gfx12) | ((3 - gfx12) << 16);
    }
    if (test.wide_window_scissor)
      state.context_registers[0x91] = 0x3fff3fff;
    if (test.positions) {
      state.context_registers[gfx12 ? 0x10b : 0x2fa] =
          state.context_registers[gfx12 ? 0x10d : 0x2fc] = std::bit_cast<uint32_t>(test.guard);
    }
    const float extent = test.extent;
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    for (uint32_t i = 0; i < 3; ++i) {
      const float w = vertex_parameters ? float(1u << i) : 1.0f;
      std::array<uint32_t, 4> position{
          std::bit_cast<uint32_t>((i == 2 ? extent : -extent) * w),
          std::bit_cast<uint32_t>((i == 1 ? extent : -extent) * w),
          std::bit_cast<uint32_t>(!test.first_vertex_depth_only || i == 0 ? test.depth : 0.0f),
          std::bit_cast<uint32_t>(w)};
      if (test.positions)
        for (uint32_t c = 0; c < 4; ++c)
          position[c] = std::bit_cast<uint32_t>((*test.positions)[i][c]);
      draw->export_lane(*wave_, i, 12, 15, position);
    }
    draw->export_lane(*wave_, 0, 20, 1,
                      {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
    if (test.outcome == Outcome::Reject) {
      EXPECT_THROW(draw->advance(*access_), std::runtime_error);
      continue;
    }
    if (test.outcome == Outcome::Empty) {
      EXPECT_FALSE(draw->advance(*access_));
      EXPECT_EQ(memory_.read32(0x100000), 0);
      continue;
    }
    const auto dispatch = draw->advance(*access_);
    ASSERT_TRUE(dispatch);
    uint32_t covered_index = 0;
    bool witness_seen = false;
    for (uint32_t workgroup = 0; workgroup < dispatch->total_wgs; ++workgroup) {
      wave_->set_wg_coord(workgroup, 0, 0);
      wave_->set_graphics_stage(draw);
      draw->initialize(*wave_, workgroup, 0);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        if (!(wave_->exec() & (uint64_t{1} << lane)))
          continue;
        if (test.fragment_inputs && wave_->debug_read_vgpr(test.fragment_inputs->values.size(),
                                                           lane) == test.fragment_inputs->xy) {
          witness_seen = true;
          for (uint32_t reg = 0; reg < test.fragment_inputs->values.size(); ++reg)
            EXPECT_EQ(wave_->debug_read_vgpr(reg, lane), test.fragment_inputs->values[reg])
                << "register=" << reg << " lane=" << lane;
        }
        if (test.inputs == 0xaf28) {
          // Pull I/W, J/W, 1/W; linear I/J; position XYZW; packed XY.
          const uint32_t x = 1 + covered_index % 2, y = 1 + covered_index / 2;
          // Screen vertices are (0,0), (0,4), (4,0), with W={1,2,4}.
          // Barycentric and reciprocal-W values below are exactly representable.
          const float b1 = (y + 0.5f) / 4, b2 = (x + 0.5f) / 4;
          const float rw = 1 - b1 - b2 + b1 / 2 + b2 / 4;
          const float expected[] = {b1 / 2, b2 / 4, rw, b1, b2, x + 0.5f, y + 0.5f, 0, 1.0f / rw};
          for (uint32_t reg = 0; reg < 9; ++reg)
            EXPECT_EQ(wave_->debug_read_vgpr(reg, lane), std::bit_cast<uint32_t>(expected[reg]))
                << "register=" << reg << " lane=" << lane;
          EXPECT_EQ(wave_->debug_read_vgpr(9, lane), 0u);
          EXPECT_EQ(wave_->debug_read_vgpr(10, lane), x | (y << 16));
          for (uint32_t c = 0; c < 4; ++c)
            for (uint32_t k = 0; k < 3; ++k)
              EXPECT_EQ(
                  lds_.read32(wave_->lds_base() + (c * 3 + k) * 4),
                  test.flat
                      ? (k ? 0u : 0x7fc00000u + ((test.polygon_mode & (1u << 19)) ? 8u : 0u) + c)
                      : std::bit_cast<uint32_t>(float(10 + k * 4 + c)));
        } else if (test.inputs == 0x8402) {
          EXPECT_EQ(wave_->debug_read_vgpr(2, lane), 0u);
          const uint32_t packed = wave_->debug_read_vgpr(3, lane);
          EXPECT_GE(packed & 0xffff, 1u);
          EXPECT_LT(packed & 0xffff, 3u);
          EXPECT_GE(packed >> 16, 1u);
          EXPECT_LT(packed >> 16, 3u);
        }
        ++covered_index;
      }
      // Include helper lanes in exports; only covered fragments may write.
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
        draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000, 0, 0});
    }
    EXPECT_EQ(covered_index, (test.triangle || test.viewport_scissor)
                                 ? std::popcount(test.expected_coverage)
                                 : std::popcount(test.sample_coverage));
    if (test.fragment_inputs) {
      EXPECT_TRUE(witness_seen);
    }
    EXPECT_FALSE(draw->advance(*access_));
    for (uint32_t y = 0; y < 4; ++y)
      for (uint32_t x = 0; x < 4; ++x) {
        const auto address = gfx12 ? amdgpu::gfx12_image_offset(x, y, 4, 4, 3)
                                   : amdgpu::gfx11_image_offset(x, y, 4, 4, 26);
        ASSERT_TRUE(address);
        const bool sample_enabled = test.sample_coverage & (1u << ((x & 1) + 2 * (y & 1)));
        const bool covered = (test.triangle || test.viewport_scissor)
                                 ? (test.expected_coverage & (1u << (y * 4 + x)))
                                 : x >= 1 && x < 3 && y >= 1 && y < 3;
        EXPECT_EQ(memory_.read32(0x100000 + *address),
                  !test.depth_only && test.color_control && test.target_mask && test.shader_mask &&
                          sample_enabled && covered
                      ? 0xff0000ffu
                      : 0u)
            << x << "," << y;
      }
  }
}

TEST_P(GraphicsExportTest, DepthClippedFanCoverageAndPolygonFacing) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    const char *name;
    std::array<int32_t, 4> offsets;
    std::array<float, 3> depth;
    uint32_t coverage;
    bool overlap = false;
  };
  // Raw 4x4 framebuffer masks agree on physical RDNA3/4 for all six vertex
  // permutations. Offsets are in 1/1024 screen pixels. Some snapped fan pieces
  // have opposite winding, although the original polygon has one facing.
  const Case cases[] = {
      {"near original", {3, 5, 5, 3}, {-1, 1, 1}, 0x8400},
      {"near B.x below", {2, 5, 5, 3}, {-1, 1, 1}, 0x8400},
      {"near B.x above", {4, 5, 5, 3}, {-1, 1, 1}, 0x8400},
      {"near B.y below", {3, 4, 5, 3}, {-1, 1, 1}, 0x8400},
      {"near B.y above", {3, 6, 5, 3}, {-1, 1, 1}, 0x8400},
      {"near C.x below", {3, 5, 4, 3}, {-1, 1, 1}, 0},
      {"near C.x above", {3, 5, 6, 3}, {-1, 1, 1}, 0x400},
      {"near C.y below", {3, 5, 5, 2}, {-1, 1, 1}, 0x400},
      {"near C.y above", {3, 5, 5, 4}, {-1, 1, 1}, 0x8400},
      {"near nonzero original area", {3, 5, 9, 3}, {-1, 1, 1}, 0x400},
      {"near collinear clipped polygon", {0, 0, 0, 0}, {-1, 1, 1}, 0},
      {"near offset control", {3, 7, 7, 3}, {-1, 1, 1}, 0x8400},
      {"far original", {3, 5, 5, 3}, {2, 0, 0}, 0x8400},
      {"unclipped collinear control", {3, 5, 5, 3}, {0, 0, 0}, 0},
      // Additive blending on both cards confirms two invocations at (2,2)
      // for cyclic orders and none for the three reversed orders.
      {"overlapping snapped fan", {-11, -13, -13, -20}, {-1, 1, 1}, 0x400, true},
  };
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  struct Restore {
    std::fenv_t &saved;
    ~Restore() { std::fesetenv(&saved); }
  } restore{saved};
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  for (uint32_t test_index = 0; test_index < std::size(cases); ++test_index) {
    const auto &test = cases[test_index];
    std::array<uint32_t, 3> permutation{0, 1, 2};
    do {
      // Only the original case has physical interior-scissor and facing/cull
      // controls. The remaining cases use the captured full-scissor mask.
      for (uint32_t control = 0; control < (test_index == 0 ? 4u : 1u); ++control) {
        SCOPED_TRACE(testing::Message()
                     << test.name << " permutation=" << permutation[0] << permutation[1]
                     << permutation[2] << " control=" << control);
        const bool interior = control == 1;
        const uint32_t cull = control < 2 ? 0 : control - 1;
        const bool expected_front =
            !((permutation[0] > permutation[1]) ^ (permutation[0] > permutation[2]) ^
              (permutation[1] > permutation[2]));
        uint32_t expected = interior ? 0x400 : test.coverage;
        if (test.overlap && !expected_front)
          expected = 0;
        if ((cull == 1 && expected_front) || (cull == 2 && !expected_front))
          expected = 0;
        const uint32_t color = cull && !expected_front ? 0xff00ff00 : 0xff0000ff;
        for (uint32_t y = 0; y < 4; ++y)
          for (uint32_t x = 0; x < 4; ++x) {
            const auto offset = gfx12 ? amdgpu::gfx12_image_offset(x, y, 4, 4, 3)
                                      : amdgpu::gfx11_image_offset(x, y, 4, 4, 26);
            ASSERT_TRUE(offset);
            memory_.write32(0x100000 + *offset, 0);
          }
        amdgpu::Pm4QueueState state;
        state.num_instances = 1;
        state.uconfig_registers[0x242] = 4;
        auto &context = state.context_registers;
        context[0x318] = 0x1000;
        context[gfx12 ? 0x3b0 : 0x31c] = 10;
        context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? (3 << 16) | 3 : (3 << 14) | 3;
        context[gfx12 ? 0x31f : 0x3b8] = gfx12 ? 3 << 15 : 26 << 14;
        context[gfx12 ? 0x214 : 0x8e] = context[gfx12 ? 0x215 : 0x8f] = 15;
        context[gfx12 ? 0x195 : 0x1c5] = 9;
        context[gfx12 ? 0x198 : 0x1b4] = 1u << 15;
        context[gfx12 ? 0x205 : 0x206] = 0x43f;
        context[gfx12 ? 0x207 : 0x205] = cull;
        context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
        context[0x204] = 1u << 19;
        context[0x2f9] = 0x2d;
        context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
            std::bit_cast<uint32_t>(2.0f);
        context[gfx12 ? 0x10b : 0x2fa] = context[gfx12 ? 0x10d : 0x2fc] =
            std::bit_cast<uint32_t>(16382.5f);
        context[0x90] = interior ? 1 | (1 << 16) : 0;
        const uint32_t limit = (interior ? 3 : 4) - gfx12;
        context[0x91] = limit | (limit << 16);
        context[0x30e] = context[0x30f] = 0x10001;
        const std::array<std::array<float, 4>, 3> positions{
            {{-1, -1, test.depth[0], 1},
             {0.5f + test.offsets[0] / 2048.0f, 0.5f + test.offsets[1] / 2048.0f, test.depth[1], 1},
             {1.5f + test.offsets[2] / 2048.0f, 1.5f + test.offsets[3] / 2048.0f, test.depth[2],
              1}}};
        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        for (uint32_t lane = 0; lane < 3; ++lane) {
          std::array<uint32_t, 4> words;
          for (uint32_t component = 0; component < 4; ++component)
            words[component] = std::bit_cast<uint32_t>(positions[permutation[lane]][component]);
          draw->export_lane(*wave_, lane, 12, 15, words);
        }
        draw->export_lane(*wave_, 0, 20, 1,
                          {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
        const auto dispatch = draw->advance(*access_);
        EXPECT_EQ(bool(dispatch), expected != 0);
        uint32_t seen = 0, invocations = 0;
        if (dispatch) {
          for (uint32_t workgroup = 0; workgroup < dispatch->total_wgs; ++workgroup) {
            wave_->set_wg_coord(workgroup, 0, 0);
            wave_->set_graphics_stage(draw);
            draw->initialize(*wave_, workgroup, 0);
            if (test_index == 0) {
              EXPECT_EQ(amdgpu::GraphicsDrawTestAccess::front(*draw, workgroup), expected_front);
            }
            for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
              if (wave_->exec() & (uint64_t{1} << lane)) {
                const uint32_t xy = wave_->debug_read_vgpr(0, lane);
                ASSERT_LT(xy & 0xffff, 4u);
                ASSERT_LT(xy >> 16, 4u);
                const uint32_t sample = 1u << ((xy >> 16) * 4 + (xy & 0xffff));
                ++invocations;
                seen |= sample;
              }
              draw->export_lane(*wave_, lane, 0, 15,
                                {color == 0xff0000ff ? 0x3f800000u : 0u,
                                 color == 0xff00ff00 ? 0x3f800000u : 0u, 0, 0x3f800000u});
            }
          }
          EXPECT_FALSE(draw->advance(*access_));
        }
        EXPECT_EQ(seen, expected);
        EXPECT_EQ(invocations, test.overlap ? (expected ? 2u : 0u) : std::popcount(expected));
        for (uint32_t y = 0; y < 4; ++y)
          for (uint32_t x = 0; x < 4; ++x) {
            const auto offset = gfx12 ? amdgpu::gfx12_image_offset(x, y, 4, 4, 3)
                                      : amdgpu::gfx11_image_offset(x, y, 4, 4, 26);
            ASSERT_TRUE(offset);
            EXPECT_EQ(memory_.read32(0x100000 + *offset),
                      expected & (1u << (y * 4 + x)) ? color : 0);
          }
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }
}

TEST_P(GraphicsExportTest, ArrayAttachmentViewsSelectProvokingVertexAndPreserveOtherLayers) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto image_address = gfx12 ? amdgpu::gfx12_image_address : amdgpu::gfx11_image_address;
  for (uint32_t target : {0u, 7u})
    for (uint32_t first : {0u, 2u})
      for (bool enabled : {false, true})
        for (bool last_provoking : {false, true})
          for (uint32_t exported : {1u, 3u}) {
            SCOPED_TRACE(testing::Message()
                         << "target=" << target << ", first=" << first << ", enabled=" << enabled
                         << ", last_provoking=" << last_provoking << ", exported=" << exported);
            amdgpu::Pm4QueueState state;
            state.num_instances = 1;
            state.uconfig_registers[0x242] = 4;
            auto &context = state.context_registers;

            const uint32_t block = 0x318 + (gfx12 ? 9 : 15) * target;
            context[gfx12 ? 0x3b0 + target : block + 4] = 10;
            context[gfx12 ? block + 6 : 0x3b0 + target] = gfx12 ? 15 | (15 << 16) : 15 | (15 << 14);
            context[gfx12 ? block + 7 : 0x3b8 + target] = (gfx12 ? 3u << 15 : 26u << 14) | 4;
            context[gfx12 ? block + 1 : block + 3] = first | ((first + 2) << (gfx12 ? 14 : 13));
            context[block] = 0x1000;
            context[gfx12 ? 0x214 : 0x8e] = 15u << (4 * target);
            context[gfx12 ? 0x215 : 0x8f] = 15u << (4 * target);
            context[gfx12 ? 0x195 : 0x1c5] = 9;
            context[gfx12 ? 0x198 : 0x1b4] = 2 | (1u << 13);
            context[0x2f9] = 0x2d;
            context[gfx12 ? 0x205 : 0x206] = 0x43f;
            context[gfx12 ? 0x206 : 0x207] = enabled ? 1u << 18 : 0;
            context[gfx12 ? 0x207 : 0x205] = last_provoking ? 1u << 19 : 0;
            context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
            context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
                std::bit_cast<uint32_t>(8.0f);
            context[0x91] = gfx12 ? 15 | (15 << 16) : 16 | (16 << 16);
            context[0x30e] = context[0x30f] = 0xffffffff;
            // One 64KiB block per slice on both chosen layouts; inspect every texel.
            for (uint32_t layer = 0; layer < 5; ++layer)
              for (uint32_t y = 0; y < 16; ++y)
                for (uint32_t x = 0; x < 16; ++x) {
                  const uint64_t base =
                      amdgpu::image_layer_base(gfx12, 0x100000, 65536, layer, 4, gfx12 ? 3 : 26);
                  const auto address = image_address(base, x, y, 16, 4, gfx12 ? 3 : 26);
                  memory_.write32(*address, 0xdeadbeef);
                }

            auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
            for (uint32_t i = 0; i < 3; ++i) {
              draw->export_lane(*wave_, i, 12, 15,
                                {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                                 std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                                 std::bit_cast<uint32_t>(1.0f)});
              draw->export_lane(*wave_, i, 13, enabled ? 15 : 4,
                                {0xdeadbeef, 0xdeadbeef, i == 2 ? exported : 0, 0xdeadbeef});
            }
            ASSERT_FALSE(wave_->instruction_execution_failed());
            draw->export_lane(*wave_, 0, 20, 1,
                              {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
            const uint32_t relative_layer = enabled && last_provoking ? exported : 0;
            auto dispatch = draw->advance(*access_);
            if (relative_layer > 2)
              EXPECT_FALSE(dispatch);
            else {
              ASSERT_TRUE(dispatch);
              EXPECT_GT(dispatch->grid_wgs_x, 1u);
              for (uint32_t workgroup = 0; workgroup < dispatch->grid_wgs_x; ++workgroup) {
                wave_->set_wg_coord(workgroup, 0, 0);
                wave_->set_graphics_stage(draw);
                draw->initialize(*wave_, workgroup, 0);
                EXPECT_EQ(wave_->debug_read_vgpr(2, 0), relative_layer << 16);
                for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
                  draw->export_lane(*wave_, lane, 0, 15, {0x3f800000, 0, 0, 0x3f800000});
              }
              EXPECT_FALSE(draw->advance(*access_));
            }
            uint32_t changed = 0;
            for (uint32_t layer = 0; layer < 5; ++layer)
              for (uint32_t y = 0; y < 16; ++y)
                for (uint32_t x = 0; x < 16; ++x) {
                  const uint64_t base =
                      amdgpu::image_layer_base(gfx12, 0x100000, 65536, layer, 4, gfx12 ? 3 : 26);
                  const auto address = image_address(base, x, y, 16, 4, gfx12 ? 3 : 26);
                  const auto value = memory_.read32(*address);
                  if (value != 0xdeadbeef) {
                    ++changed;
                    EXPECT_EQ(layer, first + relative_layer);
                    EXPECT_EQ(value, 0xff0000ffu);
                  }
                }
            if (relative_layer <= 2)
              EXPECT_GT(changed, 32u);
            else
              EXPECT_EQ(changed, 0u);
          }
}

TEST_P(GraphicsExportTest, IndexedDrawPreservesVertexIndicesAndLocalConnectivity) {
  amdgpu::Pm4QueueState state;
  state.num_instances = 1;
  state.uconfig_registers[0x242] = 4;
  amdgpu::GraphicsDraw draw(state, GetParam(), 3, {9, 4, 9});
  draw.initialize(*wave_, 0, 0);
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t vertex_id = gfx12 ? 3 : 5;
  EXPECT_EQ(wave_->debug_read_vgpr(vertex_id, 0), 9);
  EXPECT_EQ(wave_->debug_read_vgpr(vertex_id, 1), 4);
  EXPECT_EQ(wave_->debug_read_vgpr(vertex_id, 2), 9);
  const uint32_t bits = gfx12 ? 9 : 10;
  EXPECT_EQ(wave_->debug_read_vgpr(0, 0), (1u << bits) | (2u << (2 * bits)));
  state.uconfig_registers[0x24b] = 1;
  EXPECT_NO_THROW(amdgpu::GraphicsDraw(state, GetParam(), 3, {9, 4, 9}));
  // Odd strip triangles retain the API-selected first or last provoking vertex.
  state.uconfig_registers[0x242] = 6;
  state.uconfig_registers[0x24b] = 0;
  for (bool last_provoking : {false, true}) {
    state.context_registers[gfx12 ? 0x207 : 0x205] = last_provoking ? 1u << 19 : 0;
    amdgpu::GraphicsDraw strip(state, GetParam(), 4, {3, 4, 5, 6});
    strip.initialize(*wave_, 0, 0);
    EXPECT_EQ(wave_->debug_read_vgpr(0, 1), last_provoking
                                                ? 2u | (1u << bits) | (3u << (2 * bits))
                                                : 1u | (3u << bits) | (2u << (2 * bits)));
  }
}

TEST_P(GraphicsExportTest, PrimitiveRestartResetsStripWindingAndDropsIncompleteSegments) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool last_provoking : {false, true})
    for (uint32_t primitive : {4u, 6u}) {
      for (const auto &[index_type, marker] :
           {std::pair{0u, 0xffffu}, {1u, 0xffffffffu}, {2u, 0xffu}}) {
        amdgpu::Pm4QueueState state;
        state.uconfig_registers[0x242] = primitive;
        state.uconfig_registers[0x24b] = 1;
        state.uconfig_registers[0x243] = index_type;
        state.context_registers[0x103] = 0xffffffff;
        state.context_registers[gfx12 ? 0x207 : 0x205] = last_provoking ? 1u << 19 : 0;
        const std::vector<uint32_t> input{marker, 99, marker, 3,  4,  5,      6,  marker,
                                          marker, 10, 11,     12, 13, marker, 90, 91};
        const std::vector<uint32_t> expected =
            primitive == 4   ? std::vector<uint32_t>{3, 4, 5, 10, 11, 12}
            : last_provoking ? std::vector<uint32_t>{3, 4, 5, 5, 4, 6, 10, 11, 12, 12, 11, 13}
                             : std::vector<uint32_t>{3, 4, 5, 4, 6, 5, 10, 11, 12, 11, 13, 12};
        amdgpu::GraphicsDraw draw(state, GetParam(), input.size(), input);
        draw.initialize(*wave_, 0, 0);
        EXPECT_EQ(wave_->debug_read_sgpr(3),
                  expected.size() | ((expected.size() / 3) << 8) | (1u << 28));
        for (uint32_t i = 0; i < expected.size(); ++i)
          EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 3 : 5, i), expected[i]);
        for (uint32_t i = 0; i < expected.size() / 3; ++i) {
          const uint32_t bits = gfx12 ? 9 : 10;
          EXPECT_EQ(wave_->debug_read_vgpr(0, i),
                    (3 * i) | ((3 * i + 1) << bits) | ((3 * i + 2) << (2 * bits)));
        }
        // MATCH_ALL_BITS keeps upper register bits significant for narrow indices.
        state.uconfig_registers[0x24b] |= 2;
        amdgpu::GraphicsDraw full_match(state, GetParam(), input.size(), input);
        full_match.initialize(*wave_, 0, 0);
        if (marker != 0xffffffff) {
          for (uint32_t i = 0; i < input.size(); ++i)
            EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 3 : 5, i), input[i]);
        }
      }
    }
}

TEST_P(GraphicsExportTest, GeometryLaunchAndAllocationSeparateInputAndOutputCounts) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto state = batch_state(64);
  state.uconfig_registers[0x242] = 1; // Point input, triangle output.
  state.uconfig_registers[0x266] = 2;
  state.context_registers[gfx12 ? 0x2a6 : 0x2d5] |= 1u << 5;
  state.context_registers[0x2ce] = 4;        // Four possible outputs per input point.
  state.context_registers[0x204] = 1u << 22; // No rasterization in this launch test.
  auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 17);
  wave_->set_graphics_stage(draw);
  for (uint32_t group = 0; group < 2; ++group) {
    draw->initialize(*wave_, 0, 0);
    const uint32_t count = group ? 1 : 16;
    EXPECT_EQ(wave_->debug_read_sgpr(2), (count * 4 << 12) | (count << 22));
    EXPECT_EQ(wave_->debug_read_sgpr(3), count | (count << 8) | (1u << 28));
    for (uint32_t lane = 0; lane < count; ++lane) {
      EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 3 : 5, lane), group * 16 + lane);
      EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 1 : 2, lane), group * 16 + lane);
      EXPECT_EQ(wave_->debug_read_vgpr(0, lane),
                gfx12 ? lane | (lane << 9) | (lane << 18) : lane | (lane << 16));
    }
    // Compaction may produce fewer vertices; amplification may produce more.
    wave_->set_m0(3 | (1u << 12));
    ASSERT_TRUE(cu_->handle_sendmsg(*wave_, 9));
    for (uint32_t lane = 0; lane < 3; ++lane)
      draw->export_lane(*wave_, lane, 12, 15, {0, 0, 0, 0x3f800000});
    const uint32_t bits = gfx12 ? 9 : 10;
    draw->export_lane(*wave_, 0, 20, 1, {(1u << bits) | (2u << (2 * bits)), 0, 0, 0});
    EXPECT_EQ(draw->advance(*access_).has_value(), group == 0);
  }
  EXPECT_FALSE(wave_->instruction_execution_failed());
}

TEST_P(GraphicsExportTest, MergedVertexUserCountExcludesSystemRingPair) {
  amdgpu::Pm4QueueState state;
  state.num_instances = 1;
  state.uconfig_registers[0x242] = 4;
  state.sh_registers[0x8b] = 6 << 1;
  for (uint32_t i = 0; i < 6; ++i)
    state.sh_registers[0x8c + i] = 100 + i;
  amdgpu::GraphicsDraw draw(state, GetParam(), 3);
  draw.initialize(*wave_, 0, 0);
  for (uint32_t i = 0; i < 6; ++i)
    EXPECT_EQ(wave_->debug_read_sgpr(8 + i), 100 + i);
}

TEST_P(GraphicsExportTest, MergedVertexGroupCountsIncludeTrailingVertices) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    uint32_t primitive, vertices, wave, group_info;
  };
  for (const auto &test :
       {Case{4, 3, 32, 0x00403000}, Case{4, 4, 32, 0x00404000}, Case{4, 32, 32, 0x0281e000},
        Case{4, 65, 64, 0x0543f000}, Case{6, 4, 32, 0x00804000}, Case{6, 65, 64, 0x0f840000}}) {
    amdgpu::Pm4QueueState state;
    state.uconfig_registers[0x242] = test.primitive;
    state.context_registers[gfx12 ? 0x2a6 : 0x2d5] = test.wave == 32 ? 1u << 22 : 0;
    amdgpu::GraphicsDraw draw(state, GetParam(), test.vertices);
    draw.initialize(*wave_, 0, 0);
    // GS SGPR2: group primitive count[30:22], vertex count[20:12], ordered ID[11:0].
    EXPECT_EQ(wave_->debug_read_sgpr(2), test.group_info);
  }
}

TEST_P(GraphicsExportTest, VertexParametersIncludePrimitiveExportsInRingStride) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto backing = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  const auto handle = vm.register_address_space(0, backing, backing);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  auto state = batch_state(64);
  state.uconfig_registers[0x242] = 4;
  // Two vertex parameters and one primitive parameter; the PS reads vertex parameter one.
  if (gfx12) {
    state.sh_registers[0x31] = 1 | (1u << 5) | (1u << 11);
  } else {
    state.context_registers[0x1b1] = (1u << 1) | (1u << 8);
    state.context_registers[0x1b6] |= 1;
  }
  state.context_registers[gfx12 ? 0x199 : 0x191] = 1;
  auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 63);
  batch_wave(64, 0);
  draw->initialize(*wave_, 0, 0);
  for (uint32_t lane = 0; lane < 63; ++lane)
    draw->export_lane(*wave_, lane, 12, 15, {0, 0, 0, 0x3f800000});
  for (uint32_t primitive = 0; primitive < 21; ++primitive)
    draw->export_lane(*wave_, primitive, 20, 1, {1u << 31, 0, 0, 0});
  export_rectangle_vertices(*draw, 20);
  const std::array<float, 4> values{0.25f, 0.5f, 0.75f, 1.0f};
  for (uint32_t vertex = 60; vertex < 63; ++vertex) {
    // Second 32-vertex block, parameter one, with all three slots in the block stride.
    const uint64_t address = 0x10000 + 32 * 48 + 32 * 16 + (vertex - 32) * 16;
    ASSERT_EQ(access->write(address, std::as_bytes(std::span{values})),
              amdgpu::VmAccessOutcome::Complete);
  }
  ASSERT_TRUE(draw->advance(*access));
  batch_wave(32, 0);
  initialize_fragment(draw);
  for (uint32_t component = 0; component < 4; ++component) {
    EXPECT_EQ(lds_.read32(wave_->lds_base() + component * 12),
              std::bit_cast<uint32_t>(values[component]));
    EXPECT_EQ(lds_.read32(wave_->lds_base() + component * 12 + 4), 0u);
    EXPECT_EQ(lds_.read32(wave_->lds_base() + component * 12 + 8), 0u);
  }
}

TEST_P(GraphicsExportTest, AttributeWordGatherMatchesSelectedBitsAndHostEnvironment) {
#if defined(__GLIBC__) && defined(__x86_64__)
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  KfdProcess process(7);
  std::vector<uint8_t> ring(65536);
  constexpr uint64_t base = 0x10000;
  constexpr auto sealed = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
  process.map_pages(base, ring.data(), ring.size(), amdgpu::Mtype::RW, sealed);
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter(vm, &memory_);
  const auto handle = adapter.register_address_space(
      7, {.page_table = &process.page_table_,
          .page_table_mutex = &process.page_table_mutex_,
          .page_table_generation = process.page_table_generation(),
          .request_mutex = process.page_table_request_mutex(),
          .mutation_epoch = process.page_table_mutation_epoch(),
          .page_table_cache_state = process.page_table_cache_state()});
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  constexpr uint32_t values[] = {0,          0x80000000, 1,          0x007fffff, 0x3f000001,
                                 0x7f800000, 0xff800000, 0x7fc01234, 0x7f801234, 0xbf800000};
  for (size_t i = 0; i < ring.size(); i += 4)
    std::memcpy(ring.data() + i, &values[(i / 4) % std::size(values)], 4);
  std::fenv_t saved;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (uint32_t attributes : {1u, 2u, 32u}) {
    for (uint32_t stride : {16u, 48u, 512u}) {
      const uint32_t primitive_attributes = stride == 48 ? 1 : 0;
      const uint32_t vertex_attributes = stride / 16 - primitive_attributes;
      for (uint32_t offset : {0u, 32768u}) {
        auto state = batch_state(64);
        state.uconfig_registers[0x242] = 6; // A full 64-vertex triangle-strip group.
        if (gfx12) {
          state.sh_registers[0x31] =
              (attributes << 11) | (primitive_attributes << 5) | (vertex_attributes - 1);
        } else {
          state.context_registers[0x1b1] =
              ((vertex_attributes - 1) << 1) | (primitive_attributes << 8);
          state.context_registers[0x1b6] = (1u << 15) | attributes;
        }
        const uint32_t controls[] = {0, 0x400 | 31, 0x420 | 31, 0x320, 31};
        for (uint32_t a = 0; a < attributes; ++a)
          state.context_registers[(gfx12 ? 0x199 : 0x191) + a] = controls[a % std::size(controls)];
        amdgpu::GraphicsDraw draw(state, GetParam(), 64);
        constexpr std::array<uint32_t, 3> indices{63, 0, 31};
        constexpr uint32_t provoking = 32;
        std::vector<uint32_t> expected(attributes * 12), actual(attributes * 12);
        for (uint32_t a = 0; a < attributes; ++a) {
          const uint32_t control = controls[a % std::size(controls)];
          const bool passthrough = (control & 0x420) == 0x420;
          const bool flat = (control & 0x400) && !passthrough;
          for (uint32_t c = 0; c < 4; ++c)
            for (uint32_t k = 0; k < 3; ++k) {
              auto &word = expected[a * 12 + c * 3 + k];
              if ((control & 32) && !passthrough) {
                word = ((control >> 8) & (c == 3 ? 1u : 2u)) ? 0x3f800000 : 0;
              } else {
                const auto address = amdgpu::addr_calc::rdna_buffer_address(
                    3u << 30, 2u << 21, stride, true, flat ? provoking : indices[k],
                    (control & 31) * 16 + c * 4, 0, 0);
                ASSERT_EQ(access->read(base + offset + address.offset,
                                       {reinterpret_cast<std::byte *>(&word), 4}),
                          amdgpu::VmAccessOutcome::Complete);
              }
            }
        }
        for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          for (uint32_t modifiers : {0u, 0x8040u}) {
            ASSERT_EQ(std::fesetround(rounding), 0);
            ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
            std::fenv_t before;
            ASSERT_EQ(std::fegetenv(&before), 0);
            before.__mxcsr = (before.__mxcsr & ~0x8040u) | modifiers;
            ASSERT_EQ(std::fesetenv(&before), 0);
            errno = EDOM;
            const bool gathered = amdgpu::GraphicsDrawTestAccess::gather_attributes(
                draw, *access, indices, provoking, actual, offset);
            const int error = errno;
            std::fenv_t after;
            ASSERT_EQ(std::fegetenv(&after), 0);
            EXPECT_TRUE(gathered);
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(error, EDOM);
            EXPECT_EQ(after.__control_word, before.__control_word);
            EXPECT_EQ(after.__status_word, before.__status_word);
            EXPECT_EQ(after.__mxcsr, before.__mxcsr);
          }
        }
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST_P(GraphicsExportTest, AttributeWordGatherRefusesWithoutFaultsAndDefaultsNeedNoBacking) {
#if defined(__GLIBC__) && defined(__x86_64__)
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Reporter final : amdgpu::MemoryFaultReporter {
    uint32_t calls = 0;
    void report_memory_fault(uint32_t, uint64_t, amdgpu::MemoryFaultCause) override { ++calls; }
  } reporter;
  std::fenv_t saved, masked;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  ASSERT_EQ(std::fegetenv(&masked), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (uint32_t variant = 0; variant < 8; ++variant) {
    SCOPED_TRACE(variant);
    KfdProcess process(7);
    std::array<uint8_t, 4096> backing{};
    constexpr auto sealed = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
    const auto owner = variant == 3   ? amdgpu::LegacyHostExtentOwner::Driver
                       : variant == 4 ? amdgpu::LegacyHostExtentOwner::Application
                                      : sealed;
    if (variant != 0) {
      process.map_pages(0x10000, backing.data(), variant == 2 ? 32 : backing.size(),
                        amdgpu::Mtype::RW, owner);
    }
    if (variant == 1) {
      process.page_table_.at(0x10).host_extents = {{backing.data(), 2, 0, sealed},
                                                   {backing.data() + 2, 4094, 2, sealed}};
    }
    amdgpu::GpuVm vm;
    amdgpu::LegacyGpuVmAdapter adapter(vm, &memory_);
    const auto handle = adapter.register_address_space(
        7, {.page_table = &process.page_table_,
            .page_table_mutex = &process.page_table_mutex_,
            .page_table_generation = process.page_table_generation(),
            .request_mutex = process.page_table_request_mutex(),
            .mutation_epoch = process.page_table_mutation_epoch(),
            .page_table_cache_state = process.page_table_cache_state(),
            .fault_reporter = &reporter});
    const auto access = vm.snapshot(handle);
    ASSERT_TRUE(access);
    auto state = batch_state();
    if (gfx12) {
      state.sh_registers[0x31] = 1u << 11;
    } else {
      state.context_registers[0x1b6] = (1u << 15) | 1u;
    }
    state.context_registers[gfx12 ? 0x199 : 0x191] = variant == 0   ? 0x320
                                                     : variant == 6 ? 0x80000000
                                                                    : 0;
    amdgpu::GraphicsDraw draw(state, GetParam(), 3);
    std::array<uint32_t, 12> words{};
    if (variant == 7) {
      ASSERT_NE(feenableexcept(FE_DIVBYZERO), -1);
    }
    errno = EDOM;
    const bool gathered = amdgpu::GraphicsDrawTestAccess::gather_attributes(
        draw, *access, {0, 1, 2}, 0, words, 0,
        variant == 5 ? std::optional<uint64_t>{UINT64_MAX - 8} : std::nullopt);
    const int error = errno;
    ASSERT_EQ(std::fesetenv(&masked), 0);
    EXPECT_FALSE(gathered);
    EXPECT_EQ(reporter.calls, 0u);
    EXPECT_EQ(error, EDOM);
    if (variant == 0) {
      EXPECT_TRUE(std::ranges::all_of(words, [](uint32_t word) { return word == 0; }));
      export_rectangle_vertices(draw);
      EXPECT_TRUE(draw.advance(*access, nullptr, 1, true));
      EXPECT_EQ(reporter.calls, 0u);
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST_P(GraphicsExportTest, AttributeWordGatherPreservesLateFaultFlagsAndParameterBits) {
#if defined(__GLIBC__) && defined(__x86_64__)
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Observed {
    uint64_t address;
    int error;
    uint16_t control, status;
    uint32_t mxcsr;
    bool operator==(const Observed &) const = default;
  };
  struct Reporter final : amdgpu::MemoryFaultReporter {
    std::vector<Observed> observations;
    KfdProcess *process = nullptr;
    uint8_t *repair = nullptr;
    void report_memory_fault(uint32_t, uint64_t address, amdgpu::MemoryFaultCause) override {
      const int error = errno;
      std::fenv_t environment;
      std::fegetenv(&environment);
      observations.push_back({address, error, environment.__control_word, environment.__status_word,
                              environment.__mxcsr});
      // A refused gather must release all admission before the ordinary fault
      // callback, including when that callback changes this same mapping.
      if (process)
        process->map_pages(0x10000, repair, 4096, amdgpu::Mtype::RW,
                           amdgpu::LegacyHostExtentOwner::DriverSealedRam);
    }
  };
  std::fenv_t saved, masked;
  ASSERT_EQ(std::feholdexcept(&saved), 0);
  ASSERT_EQ(std::fegetenv(&masked), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (uint32_t variant = 0; variant < 4; ++variant) {
    SCOPED_TRACE(variant);
    std::vector<Observed> expected_observations;
    std::vector<uint32_t> expected_fragments;
    std::fenv_t expected_environment;
    int expected_errno = 0;
    for (bool gathered : {false, true}) {
      KfdProcess process(7);
      std::array<uint32_t, 1024> ring;
      ring.fill(0x7f800000); // Attribute differences raise invalid before the later fault.
      Reporter reporter;
      if (variant == 1) {
        reporter.process = &process;
        reporter.repair = reinterpret_cast<uint8_t *>(ring.data());
      }
      const auto owner = variant == 3 ? amdgpu::LegacyHostExtentOwner::Application
                                      : amdgpu::LegacyHostExtentOwner::DriverSealedRam;
      process.map_pages(0x10000, reinterpret_cast<uint8_t *>(ring.data()),
                        variant == 1 ? 512 : sizeof(ring), amdgpu::Mtype::RW, owner);
      amdgpu::GpuVm vm;
      amdgpu::LegacyGpuVmAdapter adapter(vm, &memory_);
      const auto handle = adapter.register_address_space(
          7, {.page_table = &process.page_table_,
              .page_table_mutex = &process.page_table_mutex_,
              .page_table_generation = process.page_table_generation(),
              .request_mutex = process.page_table_request_mutex(),
              .mutation_epoch = process.page_table_mutation_epoch(),
              .page_table_cache_state = process.page_table_cache_state(),
              .fault_reporter = &reporter});
      const auto access = vm.snapshot(handle);
      ASSERT_TRUE(access);
      auto state = batch_state();
      if (gfx12) {
        state.sh_registers[0x31] = 2u << 11;
      } else {
        state.context_registers[0x1b6] = (1u << 15) | 2u;
      }
      state.context_registers[gfx12 ? 0x199 : 0x191] = 0;
      state.context_registers[gfx12 ? 0x19a : 0x192] = variant == 2 ? 0x80000000 : 1;
      amdgpu::GraphicsDraw draw(state, GetParam(), 3);
      export_rectangle_vertices(draw);
      ASSERT_EQ(std::fesetenv(&masked), 0);
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
      errno = EDOM;
      bool failed = false;
      try {
        EXPECT_TRUE(draw.advance(*access, nullptr, 1, gathered));
      } catch (const std::runtime_error &) {
        failed = true;
      }
      const int error = errno;
      std::fenv_t environment;
      ASSERT_EQ(std::fegetenv(&environment), 0);
      EXPECT_EQ(failed, variant == 1 || variant == 2);
      EXPECT_TRUE(environment.__mxcsr & FE_INVALID);
      if (variant == 1) {
        ASSERT_EQ(reporter.observations.size(), 1u);
        EXPECT_TRUE(reporter.observations[0].mxcsr & FE_INVALID);
      } else {
        EXPECT_TRUE(reporter.observations.empty());
      }
      const auto fragments = amdgpu::GraphicsDrawTestAccess::fragment_bits(draw);
      if (!gathered) {
        expected_observations = reporter.observations;
        expected_fragments = fragments;
        expected_environment = environment;
        expected_errno = error;
      } else {
        EXPECT_EQ(reporter.observations, expected_observations);
        EXPECT_EQ(fragments, expected_fragments);
        EXPECT_EQ(environment.__control_word, expected_environment.__control_word);
        EXPECT_EQ(environment.__status_word, expected_environment.__status_word);
        EXPECT_EQ(environment.__mxcsr, expected_environment.__mxcsr);
        EXPECT_EQ(error, expected_errno);
      }
    }
  }
#else
  GTEST_SKIP();
#endif
}

TEST_P(GraphicsExportTest, AttributeWordGatherRefusesCustomBackingAndRetainsOriginalReads) {
#if defined(__GLIBC__) && defined(__x86_64__)
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto backing = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  const auto handle = vm.register_address_space(0, backing, backing);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  auto state = batch_state();
  // This test compares every private lane field, so request LINEAR inputs too.
  state.context_registers[gfx12 ? 0x198 : 0x1b4] |= 1u << 4;
  if (gfx12) {
    state.sh_registers[0x31] = 1u << 11;
  } else {
    state.context_registers[0x1b6] = (1u << 15) | 1u;
  }
  std::vector<uint32_t> expected;
  uint64_t expected_reads = 0;
  for (bool enabled : {false, true}) {
    amdgpu::GraphicsDraw draw(state, GetParam(), 3);
    std::array<uint32_t, 12> words;
    words.fill(0x12345678);
    EXPECT_FALSE(
        amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *access, {0, 1, 2}, 0, words));
    EXPECT_TRUE(std::ranges::all_of(words, [](uint32_t word) { return word == 0x12345678; }));
    EXPECT_EQ(backing->reads, 0u);
    export_rectangle_vertices(draw);
    ASSERT_TRUE(draw.advance(*access, nullptr, 1, enabled));
    const auto fragments = amdgpu::GraphicsDrawTestAccess::fragment_bits(draw);
    if (!enabled) {
      expected = fragments;
      expected_reads = backing->reads;
      EXPECT_GT(expected_reads, 0u);
    } else {
      EXPECT_EQ(fragments, expected);
      EXPECT_EQ(backing->reads, expected_reads);
    }
    EXPECT_EQ(backing->ram_lease_requests, 0u);
    backing->reads = 0;
  }
#else
  GTEST_SKIP();
#endif
}

TEST_P(GraphicsExportTest, AttributeWordGatherRefusesReplacedRawRegistration) {
#if defined(__GLIBC__) && defined(__x86_64__)
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  KfdProcess original(7), replacement(7);
  std::array<uint32_t, 1024> old_words{}, new_words{};
  old_words.fill(0x3f800000);
  new_words.fill(0x40000000);
  constexpr auto sealed = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
  original.map_pages(0x10000, reinterpret_cast<uint8_t *>(old_words.data()), sizeof(old_words),
                     amdgpu::Mtype::RW, sealed);
  replacement.map_pages(0x10000, reinterpret_cast<uint8_t *>(new_words.data()), sizeof(new_words),
                        amdgpu::Mtype::RW, sealed);
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter(vm, &memory_);
  const auto handle = adapter.register_address_space(
      7, {.page_table = &original.page_table_,
          .page_table_mutex = &original.page_table_mutex_,
          .page_table_generation = original.page_table_generation(),
          .request_mutex = original.page_table_request_mutex(),
          .mutation_epoch = original.page_table_mutation_epoch(),
          .page_table_cache_state = original.page_table_cache_state()});
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  auto state = batch_state();
  if (gfx12) {
    state.sh_registers[0x31] = 1u << 11;
  } else {
    state.context_registers[0x1b6] = (1u << 15) | 1u;
  }
  amdgpu::GraphicsDraw draw(state, GetParam(), 3);
  std::array<uint32_t, 12> words{};
  ASSERT_TRUE(
      amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *access, {0, 1, 2}, 0, words));
  EXPECT_TRUE(std::ranges::all_of(words, [](uint32_t word) { return word == 0x3f800000; }));
  const auto space = adapter.address_space_snapshot(7);
  ASSERT_TRUE(space);
  space->register_process(7, &replacement.page_table_, &replacement.page_table_mutex_,
                          replacement.page_table_generation(),
                          replacement.page_table_request_mutex(),
                          replacement.page_table_cache_state());
  EXPECT_FALSE(
      amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *access, {0, 1, 2}, 0, words));
  EXPECT_TRUE(std::ranges::all_of(words, [](uint32_t word) { return word == 0x3f800000; }));
  uint32_t current = 0;
  EXPECT_EQ(access->read(0x10000, {reinterpret_cast<std::byte *>(&current), sizeof(current)}),
            amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(current, 0x40000000u);
#else
  GTEST_SKIP();
#endif
}

TEST_P(GraphicsExportTest, AttributeWordGatherRetainsOnlyNegativeAdmissionResults) {
#if defined(__GLIBC__) && defined(__x86_64__)
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  KfdProcess process(7);
  std::array<uint32_t, 1024> first{}, second{};
  first.fill(0x3f800000);
  second.fill(0x40000000);
  constexpr auto sealed = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
  process.map_pages(0x10000, reinterpret_cast<uint8_t *>(first.data()), sizeof(first),
                    amdgpu::Mtype::RW, sealed);
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter(vm, &memory_);
  const auto handle = adapter.register_address_space(
      7, {.page_table = &process.page_table_,
          .page_table_mutex = &process.page_table_mutex_,
          .page_table_generation = process.page_table_generation(),
          .request_mutex = process.page_table_request_mutex(),
          .mutation_epoch = process.page_table_mutation_epoch(),
          .page_table_cache_state = process.page_table_cache_state()});
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  ASSERT_TRUE(access->supports_ram_word_reads());
  auto state = batch_state();
  // This test compares every private lane field, so request LINEAR inputs too.
  state.context_registers[gfx12 ? 0x198 : 0x1b4] |= 1u << 4;
  if (gfx12) {
    state.sh_registers[0x31] = 1u << 11;
  } else {
    state.context_registers[0x1b6] = (1u << 15) | 1u;
  }
  amdgpu::GraphicsDraw draw(state, GetParam(), 3);
  std::array<uint32_t, 12> words{};
  ASSERT_TRUE(
      amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *access, {0, 1, 2}, 0, words));
  EXPECT_TRUE(std::ranges::all_of(words, [](uint32_t word) { return word == 0x3f800000; }));
  process.map_pages(0x10000, reinterpret_cast<uint8_t *>(second.data()), sizeof(second),
                    amdgpu::Mtype::RW, amdgpu::LegacyHostExtentOwner::Application);
  EXPECT_FALSE(
      amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *access, {0, 1, 2}, 0, words));
  EXPECT_TRUE(std::ranges::all_of(words, [](uint32_t word) { return word == 0x3f800000; }));
  process.map_pages(0x10000, reinterpret_cast<uint8_t *>(second.data()), sizeof(second),
                    amdgpu::Mtype::RW, sealed);
  // A later eligible mapping need not revive the optional path for this draw.
  EXPECT_FALSE(
      amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *access, {0, 1, 2}, 0, words));
  amdgpu::GraphicsDraw fresh(state, GetParam(), 3);
  ASSERT_TRUE(
      amdgpu::GraphicsDrawTestAccess::gather_attributes(fresh, *access, {0, 1, 2}, 0, words));
  EXPECT_TRUE(std::ranges::all_of(words, [](uint32_t word) { return word == 0x40000000; }));
  export_rectangle_vertices(draw);
  export_rectangle_vertices(fresh);
  ASSERT_TRUE(draw.advance(*access, nullptr, 1, true));
  ASSERT_TRUE(fresh.advance(*access, nullptr, 1, false));
  EXPECT_EQ(amdgpu::GraphicsDrawTestAccess::fragment_bits(draw),
            amdgpu::GraphicsDrawTestAccess::fragment_bits(fresh));
#else
  GTEST_SKIP();
#endif
}

TEST_P(GraphicsExportTest, AttributeWordGatherCapabilityFollowsAccessStateReplacement) {
#if defined(__GLIBC__) && defined(__x86_64__)
  class Translator final : public amdgpu::AddressSpaceTranslator {
  public:
    explicit Translator(bool supported) : AddressSpaceTranslator(supported) {}
    mutable uint32_t translations = 0, copies = 0;
    amdgpu::VmTranslationResult translate(uint64_t address, size_t size,
                                          amdgpu::VmAccessKind access) const override {
      ++translations;
      return amdgpu::IdentityAddressSpaceTranslator{}.translate(address, size, access);
    }
    bool try_read_ram_words(amdgpu::PhysicalMemoryAccess &, amdgpu::VmRamRange,
                            std::span<const amdgpu::VmRamWordRead>) const override {
      ++copies;
      return false;
    }
  };
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto backing = std::make_shared<GraphicsBatchMemory>();
  auto unsupported = std::make_shared<Translator>(false);
  amdgpu::GpuVm vm;
  const auto handle = vm.register_address_space(7, unsupported, backing);
  const auto old = vm.snapshot(handle);
  ASSERT_TRUE(old);
  EXPECT_FALSE(old->supports_ram_word_reads());
  auto state = batch_state();
  if (gfx12) {
    state.sh_registers[0x31] = 1u << 11;
  } else {
    state.context_registers[0x1b6] = (1u << 15) | 1u;
  }
  amdgpu::GraphicsDraw draw(state, GetParam(), 3);
  std::array<uint32_t, 12> words{};
  EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *old, {0, 1, 2}, 0, words));
  EXPECT_EQ(unsupported->copies, 0u);
  EXPECT_EQ(unsupported->translations, 0u);
  auto supported = std::make_shared<Translator>(true);
  ASSERT_TRUE(vm.replace_translated(handle, supported, backing));
  const auto current = vm.snapshot(handle);
  ASSERT_TRUE(current);
  EXPECT_TRUE(current->supports_ram_word_reads());
  EXPECT_FALSE(old->supports_ram_word_reads());
  EXPECT_FALSE(amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *old, {0, 1, 2}, 0, words));
  EXPECT_EQ(supported->copies, 0u);
  EXPECT_FALSE(
      amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *current, {0, 1, 2}, 0, words));
  EXPECT_EQ(supported->copies, 1u); // Capability does not grant a successful copy.
  EXPECT_FALSE(
      amdgpu::GraphicsDrawTestAccess::gather_attributes(draw, *current, {0, 1, 2}, 0, words));
  EXPECT_EQ(supported->copies, 1u); // This draw remembers the admission refusal.
  EXPECT_EQ(supported->translations, 0u);
  EXPECT_EQ(backing->reads, 0u);
#else
  GTEST_SKIP();
#endif
}

TEST_P(GraphicsExportTest, VertexBatchGateRejectsAliasesUnknownRingAndOrderedModes) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto backing = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  uint32_t faults = 0;
  auto handle =
      vm.register_address_space(0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                backing, [&](uint64_t, amdgpu::VmAccessKind) { ++faults; });
  auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  for (uint32_t mode = 0; mode < 16; ++mode) {
    if (gfx12 && mode >= 14)
      continue; // GFX11 DCC metadata.
    SCOPED_TRACE(mode);
    auto state = batch_state();
    backing->alias_page.reset();
    backing->missing_page.reset();
    backing->identity_available = true;
    switch (mode) {
    case 1:
      state.context_registers[gfx12 ? 0x2a6 : 0x2d5] |= 1u << 24;
      break;
    case 2:
      state.context_registers[gfx12 ? 0x1b : 0x203] |= 1u << 16;
      break;
    case 3:
      state.sh_registers[0xb] |= 1u << 25;
      break;
    case 4:
      state.sh_registers[0x8b] |= 1u << 6;
      break;
    case 5:
      state.uconfig_registers[0x446] = 0;
      break;
    case 6:
      backing->identity_available = false;
      break;
    case 7:
      backing->missing_page = 0x10;
      break;
    case 8:
      backing->alias_page = {0x100, 0x10};
      break; // Attachment aliases the ring.
    case 9:
      state.sh_registers[gfx12 ? 0x89 : 0xc8] = 0x100;
      break; // Code in the ring.
    case 10:
      state.context_registers[gfx12 ? 0x2a6 : 0x2d5] |= 1u << 14;
      break;
    case 11:
      state.context_registers[gfx12 ? 0x2a6 : 0x2d5] |= 1u << 5;
      state.context_registers[0x2ce] = 3;
      state.uconfig_registers[0x266] = 2;
      break;
    case 12:
      state.context_registers[gfx12 ? 0x1b : 0x203] |= 1u << 12;
      break;
    case 13:
      state.context_registers[gfx12 ? 0x1b : 0x203] = 0;
      break; // Late-Z feedback.
    case 14:
    case 15:
      state.context_registers[0x3b0] = 255 | (255 << 14);
      state.context_registers[0x3b8] = (27u << 14) | (1u << 30);
      state.context_registers[0x31e] |= 1u << 22;
      state.context_registers[0x325] = 0x1800;
      if (mode == 15)
        backing->alias_page = {0x181, 0x11}; // Later DCC page aliases a later ring slot.
      break;
    }
    amdgpu::GraphicsDraw draw(state, GetParam(), 1920);
    const bool admitted = mode == 0 || mode == 14;
    EXPECT_EQ(draw.enable_vertex_batching(*access), admitted);
    EXPECT_EQ(draw.vertex_dispatch().total_wgs, admitted ? 32u : 1u);
  }
  EXPECT_EQ(faults, 0u);
  EXPECT_EQ(backing->reads, 0u);
  EXPECT_EQ(backing->writes, 0u);
}

TEST_P(GraphicsExportTest, VertexBatchPreservesIndicesInstancesStripsAndPaddedRingSlots) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto backing = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  auto handle = vm.register_address_space(
      0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(), backing);
  auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  for (uint32_t wave_size : {32u, 64u})
    for (uint32_t primitive : {4u, 6u, 17u})
      for (uint32_t stride : {1u, 2u, 31u}) {
        const uint32_t primitive_attributes = stride == 2 ? 1 : 0;
        SCOPED_TRACE(testing::Message() << wave_size << ',' << primitive << ',' << stride);
        auto state = batch_state(wave_size);
        state.num_instances = 3;
        state.uconfig_registers[0x242] = primitive;
        state.context_registers[0x204] = 1u << 22; // No rasterization.
        // Maximum stride limits Wave64 to two groups; a smaller stride admits 32.
        if (gfx12)
          state.sh_registers[0x31] = (stride - primitive_attributes) | (primitive_attributes << 5);
        else
          state.context_registers[0x1b1] =
              ((stride - primitive_attributes) << 1) | (primitive_attributes << 8);
        const uint32_t slot_bytes = wave_size * (stride + 1) * 16;
        std::vector<uint32_t> indices(769);
        for (uint32_t i = 0; i < indices.size(); ++i)
          indices[i] = 2000 - i;
        std::vector<uint32_t> expected;
        uint32_t reference_dispatches = 0;
        for (bool batched : {false, true}) {
          amdgpu::GraphicsDraw draw(state, GetParam(), indices.size(), indices);
          if (batched) {
            ASSERT_TRUE(draw.enable_vertex_batching(*access));
          }
          std::vector<uint32_t> actual;
          uint32_t dispatches = 0;
          auto dispatch = std::optional{draw.vertex_dispatch()};
          while (dispatch) {
            ++dispatches;
            EXPECT_FALSE(draw.fragment_stage());
            if (batched && dispatches == 1) {
              EXPECT_EQ(dispatch->total_wgs, std::min(32u, 65536u / slot_bytes));
            }
            for (uint32_t group = 0; group < dispatch->total_wgs; ++group) {
              batch_wave(wave_size, group);
              draw.initialize(*wave_, group, 0);
              const uint32_t offset = wave_->debug_read_sgpr(5);
              EXPECT_EQ(offset, batched ? group * slot_bytes / 512 : 0u);
              EXPECT_LE(offset, 0x7fffu);
              EXPECT_LE((offset << 9) + slot_bytes, 65536u);
              EXPECT_EQ(wave_->debug_read_sgpr(3) >> 16,
                        1u << 12); // Wave zero, one wave per workgroup.
              const uint32_t count = wave_->debug_read_sgpr(3) & 255;
              const uint32_t primitives = (wave_->debug_read_sgpr(3) >> 8) & 255;
              actual.push_back(count);
              actual.push_back(primitives);
              for (uint32_t lane = 0; lane < count; ++lane) {
                actual.push_back(wave_->debug_read_vgpr(gfx12 ? 3 : 5, lane));
                actual.push_back(wave_->debug_read_vgpr(gfx12 ? 4 : 8, lane));
                draw.export_lane(*wave_, lane, 12, 15, {0, 0, 0, 0x3f800000});
              }
              for (uint32_t lane = 0; lane < primitives; ++lane) {
                const auto connectivity = wave_->debug_read_vgpr(0, lane);
                actual.push_back(connectivity);
                draw.export_lane(*wave_, lane, 20, 1, {connectivity, 0, 0, 0});
              }
            }
            dispatch = draw.advance(*access);
          }
          if (batched) {
            EXPECT_EQ(actual, expected);
            EXPECT_LT(dispatches, reference_dispatches);
          } else {
            expected = std::move(actual);
            reference_dispatches = dispatches;
          }
        }
        // Different virtual ring pages backed by the same host page decline.
        backing->alias_page = {0x11, 0x10};
        amdgpu::GraphicsDraw aliased(state, GetParam(), indices.size(), indices);
        EXPECT_FALSE(aliased.enable_vertex_batching(*access));
        backing->alias_page.reset();
      }
}

TEST_P(GraphicsExportTest, VertexBatchCapsGroupCountAndAttributeOffsetsForLargestSlots) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto backing = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  const auto handle = vm.register_address_space(0, backing, backing);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  auto state = batch_state(64);
  state.uconfig_registers[0x447] = 15; // One MiB accommodates 32 maximum-stride slots.
  state.context_registers[gfx12 ? 0x216 : 0x202] = 0;
  if (gfx12)
    state.sh_registers[0x31] = 31;
  else
    state.context_registers[0x1b1] = 31 << 1;
  amdgpu::GraphicsDraw draw(state, GetParam(), 4096);
  ASSERT_TRUE(draw.enable_vertex_batching(*access, UINT32_MAX));
  ASSERT_EQ(draw.vertex_dispatch().total_wgs, 32u);
  batch_wave(64, 31);
  draw.initialize(*wave_, 31, 0);
  const uint32_t offset = wave_->debug_read_sgpr(5);
  EXPECT_EQ(offset, 31u * 64u);
  EXPECT_EQ(offset & ~0x7fffu, 0u);
  EXPECT_EQ((offset << 9) + 64u * 512u, 1u << 20);
}

TEST_P(GraphicsExportTest, VertexBatchMatchesSerialHelpersParametersBlendingAndSlotReuse) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (uint32_t wave_size : {32u, 64u}) {
    SCOPED_TRACE(wave_size);
    std::vector<uint32_t> reference_inputs;
    std::vector<std::byte> reference_pixels;
    uint32_t reference_dispatches = 0;
    for (bool batched : {false, true}) {
      auto backing = std::make_shared<GraphicsBatchMemory>();
      amdgpu::GpuVm vm;
      auto handle = vm.register_address_space(
          0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(), backing);
      auto access = vm.snapshot(handle);
      ASSERT_TRUE(access);
      auto state = batch_state(wave_size);
      state.num_instances = 2;
      state.uconfig_registers[0x242] = 4; // Triangles leave helper lanes uncovered.
      if (gfx12)
        state.sh_registers[0x31] = 1u << 11;
      else
        state.context_registers[0x1b6] |= 1;
      state.context_registers[gfx12 ? 0x199 : 0x191] = 0x400;     // Flat attribute zero.
      state.context_registers[0x1e0] = (1u << 30) | 4 | (5 << 8); // Source alpha blending.
      auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 1200);
      if (batched) {
        ASSERT_TRUE(draw->enable_vertex_batching(*access));
      }
      std::vector<uint32_t> inputs;
      uint32_t dispatches = 0;
      auto dispatch = std::optional{draw->vertex_dispatch()};
      while (dispatch) {
        ++dispatches;
        if (batched && dispatches == 1) {
          ASSERT_EQ(dispatch->total_wgs, 32u);
        }
        for (uint32_t group = 0; group < dispatch->total_wgs; ++group) {
          batch_wave(dispatch->kernel_wave_size, group);
          wave_->set_graphics_stage(draw);
          draw->initialize(*wave_, group, 0);
          if (!draw->fragment_stage()) {
            const uint32_t count = wave_->debug_read_sgpr(3) & 255;
            const uint32_t primitives = (wave_->debug_read_sgpr(3) >> 8) & 255;
            const uint32_t offset = wave_->debug_read_sgpr(5) << 9;
            const uint32_t first = wave_->debug_read_vgpr(gfx12 ? 3 : 5, 0);
            const uint32_t instance = wave_->debug_read_vgpr(gfx12 ? 4 : 8, 0);
            for (uint32_t lane = 0; lane < ((count + 7) & ~7u); ++lane) {
              const auto address = amdgpu::addr_calc::rdna_buffer_address(3u << 30, 2u << 21, 16,
                                                                          true, lane, 0, 0, 0);
              const std::array<float, 4> attribute{
                  float(((first + lane) / 3 + instance * 7) % 31) / 32, 0.25f, 0.5f, 0.25f};
              ASSERT_EQ(access->write(0x10000 + offset + address.offset,
                                      std::as_bytes(std::span{attribute})),
                        amdgpu::VmAccessOutcome::Complete);
              if (lane < count)
                draw->export_lane(*wave_, lane, 12, 15,
                                  {std::bit_cast<uint32_t>(lane % 3 == 2 ? 1.0f : -1.0f),
                                   std::bit_cast<uint32_t>(lane % 3 == 1 ? 1.0f : -1.0f), 0,
                                   std::bit_cast<uint32_t>(1.0f)});
            }
            for (uint32_t lane = 0; lane < primitives; ++lane)
              draw->export_lane(*wave_, lane, 20, 1, {wave_->debug_read_vgpr(0, lane), 0, 0, 0});
          } else {
            inputs.push_back(uint32_t(wave_->exec()));
            for (uint32_t i = 0; i < 12; ++i)
              inputs.push_back(lds_.read32(wave_->lds_base() + i * 4));
            std::array<uint32_t, 4> color;
            for (uint32_t i = 0; i < 4; ++i)
              color[i] = lds_.read32(wave_->lds_base() + i * 12);
            for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
              for (uint32_t reg = 0; reg < 4; ++reg)
                inputs.push_back(wave_->debug_read_vgpr(reg, lane));
              draw->export_lane(*wave_, lane, 0, 15, color);
            }
          }
        }
        dispatch = draw->advance(*access);
      }
      const auto pixels = std::span{backing->bytes}.subspan(0x100000, 1024);
      EXPECT_FALSE(inputs.empty());
      EXPECT_TRUE(
          std::ranges::any_of(pixels, [](std::byte value) { return value != std::byte{0}; }));
      if (batched) {
        EXPECT_EQ(inputs, reference_inputs);
        EXPECT_TRUE(std::ranges::equal(pixels, reference_pixels));
        EXPECT_LT(dispatches * 3, reference_dispatches);
      } else {
        reference_inputs = std::move(inputs);
        reference_pixels.assign(pixels.begin(), pixels.end());
        reference_dispatches = dispatches;
      }
      EXPECT_FALSE(wave_->instruction_execution_failed());
    }
  }
}

TEST_P(GraphicsExportTest, VertexBatchAttributeFailureUsesOrdinaryVmFaultDelivery) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto backing = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  std::vector<uint64_t> faults;
  auto handle = vm.register_address_space(0, backing, backing,
                                          [&](uint64_t address, amdgpu::VmAccessKind kind) {
                                            EXPECT_EQ(kind, amdgpu::VmAccessKind::Read);
                                            faults.push_back(address);
                                          });
  auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  auto state = batch_state();
  if (gfx12)
    state.sh_registers[0x31] = 31 | (1u << 11);
  else {
    state.context_registers[0x1b1] = 31 << 1;
    state.context_registers[0x1b6] |= 1;
  }
  auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 60);
  ASSERT_TRUE(draw->enable_vertex_batching(*access));
  ASSERT_EQ(draw->vertex_dispatch().total_wgs, 2u);
  for (uint32_t group = 0; group < 2; ++group) {
    batch_wave(32, group);
    draw->initialize(*wave_, group, 0);
    for (uint32_t lane = 0; lane < 30; ++lane)
      draw->export_lane(*wave_, lane, 12, 15,
                        {std::bit_cast<uint32_t>(lane % 3 == 2 ? 1.0f : -1.0f),
                         std::bit_cast<uint32_t>(lane % 3 == 1 ? 1.0f : -1.0f), 0, 0x3f800000});
    for (uint32_t lane = 0; lane < 10; ++lane)
      draw->export_lane(*wave_, lane, 20, 1, {wave_->debug_read_vgpr(0, lane), 0, 0, 0});
  }
  // A later mapping failure is handled at the real access, not by retained host pointers.
  backing->missing_page = 0x14;
  EXPECT_THROW(draw->advance(*access), std::runtime_error);
  EXPECT_EQ(faults, (std::vector<uint64_t>{0x14000}));
  EXPECT_GT(backing->reads, 1u);
  EXPECT_EQ(backing->writes, 0u);
}

TEST_P(GraphicsExportTest, VertexBatchDrainsBoundedFragmentWindowsBeforeReusingSlots) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto backing = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  const auto handle = vm.register_address_space(0, backing, backing);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  auto state = batch_state();
  auto &ctx = state.context_registers;
  ctx[gfx12 ? 0x216 : 0x202] = 0; // No attachment IO: exercise scheduling and coverage.
  ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(32.0f);
  ctx[0x91] = (64 - gfx12) | ((64 - gfx12) << 16);
  amdgpu::GraphicsDraw draw(state, GetParam(), 1050);
  ASSERT_TRUE(draw.enable_vertex_batching(*access));
  uint32_t vertex_dispatches = 0, fragment_dispatches = 0, waves = 0;
  auto dispatch = std::optional{draw.vertex_dispatch()};
  while (dispatch) {
    if (!draw.fragment_stage()) {
      ++vertex_dispatches;
      ASSERT_EQ(dispatch->total_wgs, vertex_dispatches == 1 ? 32u : 3u);
      EXPECT_EQ(fragment_dispatches, vertex_dispatches == 1 ? 0u : 8u);
      for (uint32_t group = 0; group < dispatch->total_wgs; ++group) {
        batch_wave(32, group);
        draw.initialize(*wave_, group, 0);
        EXPECT_EQ(wave_->debug_read_sgpr(5), group);
        for (uint32_t lane = 0; lane < 30; ++lane)
          draw.export_lane(*wave_, lane, 12, 15,
                           {std::bit_cast<uint32_t>(lane % 3 == 2 ? 1.0f : -1.0f),
                            std::bit_cast<uint32_t>(lane % 3 == 1 ? 1.0f : -1.0f), 0, 0x3f800000});
        for (uint32_t lane = 0; lane < 10; ++lane)
          draw.export_lane(*wave_, lane, 20, 1, {wave_->debug_read_vgpr(0, lane), 0, 0, 0});
      }
    } else {
      ++fragment_dispatches;
      waves += dispatch->total_wgs;
      EXPECT_LE(dispatch->total_wgs, 4096u + 1280u);
      // No output or side effect is enabled; an empty FS need not export a value.
    }
    dispatch = draw.advance(*access);
  }
  EXPECT_EQ(vertex_dispatches, 2u);
  EXPECT_EQ(fragment_dispatches, 9u);
  EXPECT_EQ(waves, 350u * 4096u / 32u);
  EXPECT_EQ(backing->writes, 0u);
}

TEST_P(GraphicsExportTest, TriangleAndRectangleListsKeepTrailingVerticesWithoutCreatingPrimitives) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (uint32_t primitive : {4u, 17u})
    for (const auto &[vertices, instances] :
         {std::pair{4u, 2u}, {5u, 2u}, {31u, 2u}, {32u, 2u}, {64u, 2u}, {65u, 2u}, {3u, 8192u}}) {
      SCOPED_TRACE(testing::Message() << "vertices=" << vertices << ", primitive="
                                      << (primitive == 4 ? "triangle list" : "rectangle list"));
      amdgpu::Pm4QueueState state;
      state.num_instances = instances;
      state.sh_registers[0x8b] = (gfx12 ? 1u : 3u) << 16;
      state.uconfig_registers[0x242] = primitive;
      auto &context = state.context_registers;
      context[gfx12 ? 0x2a6 : 0x2d5] = 1u << 22; // Wave32: groups of thirty vertices.
      context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
      context[0x2f9] = 0x2d;
      context[gfx12 ? 0x205 : 0x206] = 0x43f;
      context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
      context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
          std::bit_cast<uint32_t>(2.0f);
      context[gfx12 ? 0x3b0 : 0x31c] = 10;
      context[0x318] = 0x1000;
      context[gfx12 ? 0x214 : 0x8e] = 15;
      context[gfx12 ? 0x215 : 0x8f] = 15;
      context[gfx12 ? 0x195 : 0x1c5] = 9;
      // Cull these primitives so advance returns the
      // next vertex group, allowing us to inspect every invocation and primitive.
      context[gfx12 ? 0x207 : 0x205] = 3;

      auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), vertices);
      uint32_t invocations = 0, primitives = 0, groups = 0;
      do {
        ASSERT_LT(groups++, instances * ((vertices + 29) / 30));
        draw->initialize(*wave_, 0, 0);
        const uint32_t count = wave_->debug_read_sgpr(3) & 255;
        const uint32_t primitive_count = (wave_->debug_read_sgpr(3) >> 8) & 255;
        for (uint32_t i = 0; i < count; ++i) {
          EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 3 : 5, i), invocations % vertices);
          EXPECT_EQ(wave_->debug_read_vgpr(gfx12 ? 4 : 8, i), invocations / vertices);
          ++invocations;
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i % 3 == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i % 3 == 1 ? 1.0f : -1.0f), 0,
                             std::bit_cast<uint32_t>(1.0f)});
        }
        for (uint32_t i = 0; i < primitive_count; ++i)
          draw->export_lane(*wave_, i, 20, 1, {wave_->debug_read_vgpr(0, i), 0, 0, 0});
        primitives += primitive_count;
      } while (draw->advance(*access_));
      EXPECT_EQ(invocations, instances * vertices);
      EXPECT_EQ(primitives, instances * (vertices / 3));
      EXPECT_FALSE(wave_->instruction_execution_failed());
    }
}

TEST_P(GraphicsExportTest, BcScalarLoadsPreserveFractionalInterpolationAndNormalization) {
  // Physical GFX1100 and GFX1201 captures. The UNORM witness is one ULP
  // below exact division by 255; BC4/5 retain the six interpolation bits.
  struct Case {
    uint32_t format, texel, expected;
    std::array<uint8_t, 8> block;
  };
  for (const auto &test :
       {Case{115, 4, 0x3da08080, {0x10, 0x15, 0x94, 0x58, 0x50, 0x36, 0xbb, 0x53}},
        Case{116, 1, 0xbd614285, {0x61, 0xe8, 0x3e, 0x23, 0xc6, 0x4c, 0xef, 0x42}}}) {
    amdgpu::VectorMemState d(amdgpu::GLOBAL_MEM);
    d.wf_size = wave_->wf_size();
    d.exec_mask = d.lane_mask = 1;
    d.elem_size = 8;
    d.buffer_components = 4;
    d.buffer_selectors = 0x924; // Repeat the only decoded channel.
    d.decoded_buffer_format = amdgpu::decode_buffer_format(22).value();
    d.image_bc_format = test.format;
    d.image_bc_texels[0] = test.texel;
    d.dst_reg_base = wave_->vgpr_alloc().base + 12;
    d.response_data.resize(d.wf_size * d.elem_size);
    std::copy(test.block.begin(), test.block.end(), d.response_data.begin());
    amdgpu::complete_buffer_format_load(*wave_, *cu_, d);
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), test.expected);
  }
}

TEST_P(GraphicsExportTest, BcScalarFilteringMatchesPhysicalPrecisionAndMipOrder) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    uint32_t format, mode, sample;
    std::array<uint32_t, 2> expected;
    std::optional<uint32_t> gfx12_green = std::nullopt;
  };
  // Physical GFX1100/GFX1201 witnesses: signed floor, bilinear rounding carry,
  // weighted-mip precision, cancellation, and the architecture's mip order.
  // Modes select bilinear, explicit fractional LOD, or anisotropic sampling.
  constexpr Case cases[] = {
      {115, 0, 775, {0x3f0d33cc, 0x3f0d33cc}},
      {116, 0, 6, {0xbd484081, 0xbd484081}},
      {117, 1, 23, {0x3ee48dce, 0x3f1d9020}},
      {118, 1, 45, {0x3eb6cc38, 0xbec10c18}},
      {116, 2, 144191, {0x3ca6870e, 0x3ca6870e}},
      {117, 2, 254703, {0x3ede7dbe, 0x3f0d856d}, 0x3f0d8575},
      {118, 2, 225003, {0xbe2cfefe, 0xbed1aab5}},
  };
  wave_->set_exec(1);
  for (const auto &test : cases) {
    SCOPED_TRACE(test.format);
    SCOPED_TRACE(test.sample);
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    const uint32_t stride = test.mode ? 512 : 256;
    const uint32_t block = test.sample / 1024;
    std::array<uint8_t, 512> bytes{};
    uint32_t seed = 0xdead1234;
    for (uint32_t i = 0; i < (block + 1) * stride; ++i) {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      if (i >= block * stride)
        bytes[i - block * stride] = seed;
    }
    ASSERT_EQ(access_->write(0x100000, std::as_bytes(std::span(bytes))),
              amdgpu::VmAccessOutcome::Complete);
    // The 2x2 mip precedes the 4x4 base level in a two-level linear image.
    const uint32_t mips = test.mode != 0;
    const std::array<uint32_t, 8> descriptor{
        0x1000,   (test.format << (gfx12 ? 17 : 20)) | (3u << 30) | (mips << (gfx12 ? 12 : 16)),
        3u << 14, (9u << 28) | 0xfac | (mips << (gfx12 ? 15 : 16)),
        0,        0,
        0,        0};
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->debug_write_sgpr(4, test.mode == 2 ? 0x08000892 : 0x08000092);
    wave_->debug_write_sgpr(5, (mips * 256) << (gfx12 ? 13 : 12));
    wave_->debug_write_sgpr(6, test.mode == 2 ? 0x08f00000 : 0x00500000 | (mips << 27));
    wave_->debug_write_sgpr(7, 0);
    const uint32_t phase = (test.sample / 16) % 64, lane = test.sample % 16;
    const float u = (lane % 4 + float((phase * 37 + lane * 11) & 255) / 256) / 4;
    const float v = (lane / 4 + float((phase * 71 + lane * 23) & 255) / 256) / 4;
    std::array<float, 6> operands{};
    if (test.mode == 2)
      operands = {float(1u << (lane % 4)) / 4,
                  float(phase % 3) / 8,
                  float(phase % 5) / 64,
                  (1 + float((phase / 3) % 4) / 4) / 4,
                  u,
                  v};
    else
      operands = {u, v, float((phase * 13 + lane * 47) & 255) / 256};
    for (uint32_t r = 0; r < operands.size(); ++r)
      wave_->debug_write_vgpr(r, 0, std::bit_cast<uint32_t>(operands[r]));
    ASSERT_NO_FATAL_FAILURE(sample(test.mode == 2 ? 28 : test.mode == 1 ? 29 : 31, 1));
    EXPECT_EQ(wave_->debug_read_vgpr(12, 0), test.expected[0]);
    EXPECT_EQ(wave_->debug_read_vgpr(13, 0),
              gfx12 ? test.gfx12_green.value_or(test.expected[1]) : test.expected[1]);
  }
}

TEST(GraphicsImageFormatTest, BcInterpolationMatchesRdna3AndRdna4) {
  // FNV-1a hashes of all 16 RGBA8 texels, captured independently on both GPUs.
  // The blocks cover both endpoint orders and every color/alpha selector.
  constexpr uint32_t blocks[]{0, 1, 2, 7, 43, 97};
  constexpr uint64_t expected[3][6]{
      {0xcf6d088a1193a6f5ull, 0x170408a3b8d6febfull, 0x139152c461f567bbull, 0x6aa6d72d563f2508ull,
       0xf86cd05783cecc85ull, 0x0cb5fe42e8c4286dull},
      {0x92fe703a06f93aecull, 0xb28c896239fb69d1ull, 0xeba2629498f349e3ull, 0x8402a6a0f72745b2ull,
       0xd48442e0e2485278ull, 0xc06db0f813d86760ull},
      {0xd75eabf12b7d6380ull, 0x1c37f758e16d6147ull, 0x077c2e6c2da77a3bull, 0x17a176ec70a0810eull,
       0xdeacfb6f1812a7abull, 0xfaa78b6f9b3e3b85ull}};
  std::array<uint8_t, 98 * 256> input;
  uint32_t seed = 0xdead1234;
  for (auto &byte : input) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    byte = seed;
  }
  for (uint32_t format = 109; format <= 114; ++format)
    for (uint32_t i = 0; i < std::size(blocks); ++i) {
      uint64_t hash = 0xcbf29ce484222325ull;
      const auto block = std::span(input).subspan(blocks[i] * 256, format <= 110 ? 8 : 16);
      for (uint32_t texel = 0; texel < 16; ++texel)
        for (uint8_t value : amdgpu::decode_image_bc(format, texel, block))
          hash = (hash ^ value) * 0x100000001b3ull;
      EXPECT_EQ(hash, expected[(format - 109) / 2][i]) << format << ',' << blocks[i];
    }
}

TEST(GraphicsImageAddressTest, MatchesGfx12CoordinateBitsAndCrossesTileRows) {
  using amdgpu::gfx12_image_offset;
  EXPECT_EQ(gfx12_image_offset(1, 0, 256, 4, 3), 4);
  EXPECT_EQ(gfx12_image_offset(0, 1, 256, 4, 3), 8);
  EXPECT_EQ(gfx12_image_offset(8, 0, 256, 4, 3), 512);
  EXPECT_EQ(gfx12_image_offset(0, 8, 256, 4, 3), 256);
  EXPECT_EQ(gfx12_image_offset(128, 0, 256, 4, 3), 65536);
  EXPECT_EQ(gfx12_image_offset(0, 128, 256, 4, 3), 131072);
  EXPECT_EQ(gfx12_image_offset(2, 3, 64, 4, 0), 776);
  EXPECT_FALSE(gfx12_image_offset(64, 0, 64, 4, 3));
  EXPECT_FALSE(gfx12_image_offset(0, 0, 64, 3, 3));
  EXPECT_FALSE(gfx12_image_offset(0, 0, 64, 4, 5));
  EXPECT_EQ(amdgpu::gfx12_image_address(0x140100, 0, 8, 256, 4, 4), 0x140000);
  EXPECT_EQ(amdgpu::gfx12_image_address(0x140100, 255, 255, 256, 4, 4), 0x17fefc);
  EXPECT_EQ(amdgpu::gfx12_image_address(0x140100, 256, 0, 512, 4, 4), 0x180100);
}

TEST(GraphicsImageAddressTest, MatchesGfx11AddrLibWithPipeXorAndAlignedLinearPitch) {
  // AddrLib: GFX11, GB_ADDR_CONFIG=0x545, single-level 2D surfaces.
  struct Case {
    uint32_t swizzle, bytes, width, x, y;
    uint64_t offset;
  };
  constexpr Case cases[] = {
      {0, 4, 420, 419, 319, 573324},  {0, 4, 1024, 513, 777, 3184644},
      {2, 4, 420, 419, 319, 542652},  {2, 4, 1024, 513, 777, 3194892},
      {6, 4, 420, 419, 319, 570812},  {6, 4, 1024, 513, 777, 3211532},
      {10, 4, 420, 419, 319, 734652}, {10, 4, 1024, 513, 777, 3408140},
      {22, 4, 420, 419, 319, 571836}, {22, 4, 1024, 513, 777, 3211532},
      {24, 4, 420, 419, 319, 742332}, {24, 4, 1024, 513, 777, 3426316},
      {26, 4, 420, 419, 319, 730300}, {26, 4, 1024, 513, 777, 3408140},
      {27, 4, 420, 419, 319, 742332}, {27, 4, 1024, 513, 777, 3426316},
      {28, 4, 420, 419, 319, 873404}, {28, 4, 1024, 513, 777, 3721228},
      {30, 4, 420, 419, 319, 926908}, {30, 4, 1024, 513, 777, 3670284},
      {31, 4, 420, 419, 319, 873404}, {31, 4, 1024, 513, 777, 3721228},
  };
  for (const auto &c : cases)
    EXPECT_EQ(amdgpu::gfx11_image_offset(c.x, c.y, c.width, c.bytes, c.swizzle), c.offset)
        << c.swizzle << "," << c.x << "," << c.y;
  EXPECT_FALSE(amdgpu::gfx11_image_offset(0, 0, 64, 4, 18));
  EXPECT_FALSE(amdgpu::gfx11_image_offset(0, 0, 64, 16, 24));
  EXPECT_FALSE(amdgpu::gfx11_image_offset(64, 0, 64, 4, 26));
}

TEST(GraphicsImageAddressTest, TiledEquationsMatchAddrLibAcrossTexelSizes) {
  // Mesa 25.2.8 AddrLib, GFX11 GB_ADDR_CONFIG=0x545 and GFX12, single-level 2D.
  // Each digest covers both dimensions below, the individual coordinate bits,
  // mixed coordinates, and the final pixel. Expected values come from AddrLib's
  // surface-address API, independently of the simulator's equation tables.
  struct Case {
    bool gfx12;
    uint32_t swizzle, bytes;
    uint64_t digest;
  };
  constexpr Case cases[] = {
      {false, 0, 1, 0x8d238fdaa38a0f09ull},   {false, 0, 2, 0x4b7e6185845bcd6dull},
      {false, 0, 4, 0xdbaedd5aa75497d5ull},   {false, 0, 8, 0x9343fb18ec5e8625ull},
      {false, 0, 16, 0xae97d2316c06bca5ull},  {false, 2, 1, 0x1eb5d01f12181f8dull},
      {false, 2, 2, 0x507cc28382b86cc5ull},   {false, 2, 4, 0x849c773adccced05ull},
      {false, 2, 8, 0x4f052becc8c873e5ull},   {false, 2, 16, 0xf0ff5085f0b855e5ull},
      {false, 6, 1, 0x524cea743a8d538dull},   {false, 6, 2, 0x0b8a11ac88d17ec5ull},
      {false, 6, 4, 0x3bdd40ffdcf75905ull},   {false, 6, 8, 0x5d3faf0fd5c17de5ull},
      {false, 6, 16, 0x59772612ddf16ce5ull},  {false, 10, 1, 0x5e8fe8e48384738dull},
      {false, 10, 2, 0x4ee5b8ce3c041ec5ull},  {false, 10, 4, 0x5e3b78a302d63905ull},
      {false, 10, 8, 0x109cfdbb9ce95de5ull},  {false, 10, 16, 0xc0db1fd1c5c8ece5ull},
      {false, 22, 1, 0x8a78728e37e5f28dull},  {false, 22, 2, 0xbbbf962d26c3e9c5ull},
      {false, 22, 4, 0xeaafbee485d9c805ull},  {false, 22, 8, 0x5197669050ef92e5ull},
      {false, 22, 16, 0xac24dbd8673ab6e5ull}, {false, 24, 1, 0xca319fb842fd1e8dull},
      {false, 24, 2, 0x97ae25ba9eb2adc5ull},  {false, 24, 4, 0x452aa64d03101205ull},
      {false, 24, 8, 0x68b27d70a5e83ce5ull},  {false, 26, 1, 0x4e294f3dd837808dull},
      {false, 26, 2, 0xdcd27134ae6b03c5ull},  {false, 26, 4, 0x06f82e39ddc9a405ull},
      {false, 26, 8, 0x456b3eef3ee302e5ull},  {false, 26, 16, 0xe01edf2a729f07e5ull},
      {false, 27, 1, 0xca319fb842fd1e8dull},  {false, 27, 2, 0x97ae25ba9eb2adc5ull},
      {false, 27, 4, 0x452aa64d03101205ull},  {false, 27, 8, 0x68b27d70a5e83ce5ull},
      {false, 27, 16, 0xfe0c8c305c13a1e5ull}, {false, 28, 1, 0xdcc8a89c1fdb9e8dull},
      {false, 28, 2, 0xdf9a13dea1862dc5ull},  {false, 28, 4, 0x5805defdea8e9205ull},
      {false, 28, 8, 0x0f39b3e80a0b7ce5ull},  {false, 30, 1, 0x75412891181c808dull},
      {false, 30, 2, 0x0fb06a1f0dcd03c5ull},  {false, 30, 4, 0xdf695bb409bfa405ull},
      {false, 30, 8, 0xcea90862c4cf02e5ull},  {false, 30, 16, 0xb661ad45b32f07e5ull},
      {false, 31, 1, 0xdcc8a89c1fdb9e8dull},  {false, 31, 2, 0xdf9a13dea1862dc5ull},
      {false, 31, 4, 0x5805defdea8e9205ull},  {false, 31, 8, 0x0f39b3e80a0b7ce5ull},
      {false, 31, 16, 0x393179573ac1e1e5ull}, {true, 0, 1, 0x8d238fdaa38a0f09ull},
      {true, 0, 2, 0x7761ac6476a2a1edull},    {true, 0, 4, 0xdbaedd5aa75497d5ull},
      {true, 0, 8, 0xf970a2b6d5f302a5ull},    {true, 0, 16, 0x081ea84c7988fc25ull},
      {true, 1, 1, 0x1eb5d01f12181f8dull},    {true, 1, 2, 0x507cc28382b86cc5ull},
      {true, 1, 4, 0x849c773adccced05ull},    {true, 1, 8, 0x4f052becc8c873e5ull},
      {true, 1, 16, 0xf0ff5085f0b855e5ull},   {true, 2, 1, 0x524cea743a8d538dull},
      {true, 2, 2, 0x0b8a11ac88d17ec5ull},    {true, 2, 4, 0x3bdd40ffdcf75905ull},
      {true, 2, 8, 0x5d3faf0fd5c17de5ull},    {true, 2, 16, 0x59772612ddf16ce5ull},
      {true, 3, 1, 0x5e8fe8e48384738dull},    {true, 3, 2, 0x4ee5b8ce3c041ec5ull},
      {true, 3, 4, 0x5e3b78a302d63905ull},    {true, 3, 8, 0x109cfdbb9ce95de5ull},
      {true, 3, 16, 0xc0db1fd1c5c8ece5ull},   {true, 4, 1, 0x7606f62441cd738dull},
      {true, 4, 2, 0x6272319766441ec5ull},    {true, 4, 4, 0xe36a8ae2078c3905ull},
      {true, 4, 8, 0x1404f02b7a895de5ull},    {true, 4, 16, 0x79cd39a75428ece5ull},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(testing::Message() << c.gfx12 << ',' << c.swizzle << ',' << c.bytes);
    const auto address = c.gfx12 ? amdgpu::gfx12_image_offset : amdgpu::gfx11_image_offset;
    uint64_t digest = 0xcbf29ce484222325ull;
    for (uint32_t width : {420u, 4096u}) {
      const uint32_t height = width == 420 ? 320 : 4096;
      for (uint32_t sample = 0; sample < 64; ++sample) {
        uint32_t x = 0, y = 0;
        if (sample < 12)
          x = (1u << sample) % width;
        else if (sample < 24)
          y = (1u << (sample - 12)) % height;
        else {
          x = (sample * 1387) % width;
          y = (sample * 2477) % height;
        }
        if (sample == 63) {
          x = width - 1;
          y = height - 1;
        }
        const auto offset = address(x, y, width, c.bytes, c.swizzle);
        ASSERT_TRUE(offset);
        digest = (digest ^ *offset) * 0x100000001b3ull;
      }
    }
    EXPECT_EQ(digest, c.digest);
  }
}

TEST(GraphicsImageAddressTest, MipAddressesMatchAddrLibAtOddSizesAndTailTransitions) {
  // Independent AddrLib addresses of the last accessible texel of each view.
  struct Case {
    bool gfx12;
    uint32_t swizzle, bytes, width, height, levels, level;
    uint64_t expected;
  };
  constexpr Case cases[] = {
      {false, 0, 4, 129, 33, 8, 2, 4988},
      {true, 0, 4, 129, 33, 8, 2, 4988},
      {false, 2, 8, 200, 180, 8, 2, 29960},
      {true, 1, 8, 200, 180, 8, 2, 29960},
      {false, 22, 2, 200, 180, 8, 2, 9026},
      {true, 2, 2, 200, 180, 8, 2, 11074},
      {false, 24, 1, 200, 180, 8, 3, 49508},
      {false, 28, 2, 512, 512, 10, 3, 208382},
      {false, 27, 4, 200, 180, 8, 1, 121628},
      {true, 3, 4, 200, 180, 8, 1, 124188},
      {false, 27, 4, 200, 180, 8, 2, 52868},
      {true, 3, 4, 200, 180, 8, 2, 47492},
      {false, 31, 16, 4093, 2049, 12, 4, 1011168},
      {true, 4, 16, 4093, 2049, 12, 4, 1048544},
      {false, 30, 4, 512, 512, 10, 9, 1536},
      {true, 4, 4, 512, 512, 10, 9, 1536},
      // Uncompressed block views retain the compressed image's mip count.
      {false, 2, 16, 4, 4, 5, 3, 256},
      {true, 1, 16, 4, 4, 5, 3, 256},
      {false, 2, 16, 4, 4, 5, 4, 0},
      {true, 1, 16, 4, 4, 5, 4, 0},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(testing::Message() << c.gfx12 << "," << c.swizzle << "," << c.level);
    const auto mip =
        amdgpu::image_mip_layout(c.gfx12, c.swizzle, c.bytes, c.width, c.height, c.levels, c.level);
    ASSERT_TRUE(mip);
    const auto image_address = c.gfx12 ? amdgpu::gfx12_image_address : amdgpu::gfx11_image_address;
    EXPECT_EQ(image_address(mip->offset, mip->tail_x + mip->width - 1,
                            mip->tail_y + mip->height - 1, mip->pitch, c.bytes, c.swizzle),
              c.expected);
  }
  for (bool gfx12 : {false, true}) {
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 64, 64, 0, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 64, 64, 18, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 64, 64, 7, 7));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 0, 64, 1, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 4, 65537, 64, 1, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 18, 4, 64, 64, 7, 0));
    EXPECT_FALSE(amdgpu::image_mip_layout(gfx12, 0, 3, 64, 64, 7, 0));
  }
}

TEST(GraphicsImageAddressTest, ArrayLayersUseMipChainStrideAndSliceXor) {
  struct Case {
    bool gfx12;
    uint32_t swizzle, bytes, levels, level, x, y, layer;
    uint64_t slice_size, address;
  };
  // Independent AddrLib offsets for 129x71 surfaces, including linear rows,
  // mip tails, different block sizes and layer-dependent pipe/bank selection.
  const Case cases[] = {
      {false, 0, 4, 7, 3, 12, 6, 3, 82432, 251440},
      {true, 0, 4, 7, 3, 12, 6, 3, 82432, 250672},
      {false, 22, 4, 7, 2, 19, 14, 35, 102400, 3593652},
      {false, 26, 8, 7, 2, 19, 14, 35, 393216, 13799848},
      {false, 27, 4, 7, 2, 19, 14, 3, 262144, 836788},
      {false, 27, 8, 7, 2, 19, 14, 7, 393216, 2776744},
      {false, 27, 16, 7, 2, 19, 14, 7, 655360, 4676816},
      {false, 31, 8, 7, 2, 19, 14, 31, 524288, 16428200},
      {false, 31, 16, 7, 2, 19, 14, 31, 1048576, 32562384},
      {true, 1, 4, 7, 2, 19, 14, 3, 57088, 175796},
      {true, 3, 8, 7, 2, 19, 14, 35, 393216, 13798824},
      {true, 4, 16, 7, 2, 19, 14, 255, 1048576, 267527632},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(testing::Message()
                 << test.gfx12 << ", swizzle=" << test.swizzle << ", bytes=" << test.bytes
                 << ", level=" << test.level << ", layer=" << test.layer);
    const auto mip = amdgpu::image_mip_layout(test.gfx12, test.swizzle, test.bytes, 129, 71,
                                              test.levels, test.level);
    ASSERT_TRUE(mip);
    EXPECT_EQ(mip->slice_size, test.slice_size);
    const uint64_t base = amdgpu::image_layer_base(test.gfx12, mip->offset, mip->slice_size,
                                                   test.layer, test.bytes, test.swizzle);
    const auto image_address =
        test.gfx12 ? amdgpu::gfx12_image_address : amdgpu::gfx11_image_address;
    const auto address = image_address(base, test.x + mip->tail_x, test.y + mip->tail_y, mip->pitch,
                                       test.bytes, test.swizzle);
    ASSERT_TRUE(address);
    EXPECT_EQ(*address, test.address);
  }
}

TEST(GraphicsImageAddressTest, VolumeMipAddressesMatchAddrLib) {
  // A deep volume can enter the tail before its Z extent fits one tile.
  const auto deep = amdgpu::image_volume_mip_layout(true, 5, 1, 8, 8, 1024, 4, 0);
  ASSERT_TRUE(deep);
  EXPECT_EQ(amdgpu::image_volume_address(true, 0, *deep, 0, 0, 0, 1, 5), 2048u);
  // Independent AddrLib addresses for 129x71x37 R32, eight mip levels.
  const uint64_t offsets[2][3][4] = {
      {{918604, 18896, 4176, 512}, {1040972, 70096, 34896, 2560}, {1761868, 136656, 34896, 2560}},
      {{921164, 16848, 4176, 512}, {1044044, 70096, 32848, 2560}, {1764940, 135632, 32848, 2560}}};
  const uint32_t levels[] = {0, 3, 4, 7};
  for (bool gfx12 : {false, true})
    for (uint32_t mode = 0; mode < 3; ++mode)
      for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t swizzle = gfx12 ? 5 + mode : 21 + 4 * mode;
        SCOPED_TRACE(testing::Message()
                     << "gfx12=" << gfx12 << " swizzle=" << swizzle << " level=" << levels[i]);
        const auto mip =
            amdgpu::image_volume_mip_layout(gfx12, swizzle, 4, 129, 71, 37, 8, levels[i]);
        ASSERT_TRUE(mip);
        const uint32_t x = 666 % mip->plane.width, y = 414 % mip->plane.height,
                       z = 234 % mip->depth;
        EXPECT_EQ(amdgpu::image_volume_address(gfx12, 0, *mip, x, y, z, 4, swizzle),
                  offsets[gfx12][mode][i]);
        EXPECT_EQ(amdgpu::image_volume_address(gfx12, 0x100000100ull, *mip, x, y, z, 4, swizzle),
                  0x100000000ull + (offsets[gfx12][mode][i] ^ 0x100));
        EXPECT_FALSE(amdgpu::image_volume_address(gfx12, 0, *mip, x, y, mip->depth, 4, swizzle));
      }
}

TEST_P(GraphicsExportTest, OcclusionCountsCoveredSamplesAfterShaderDiscard) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool enabled : {false, true}) {
    auto state = batch_state(64);
    state.context_registers[gfx12 ? 0x18 : 1] = enabled ? 0x11000106 : 0;
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    export_rectangle_vertices(*draw);
    ASSERT_TRUE(draw->advance(*access_));
    initialize_fragment(draw);
    draw->export_mask(*wave_, 0x5555555555555555ull);
    EXPECT_FALSE(draw->advance(*access_));
    EXPECT_EQ(draw->occlusion_samples(), enabled ? 8u : 0u);
  }
}

TEST_P(GraphicsExportTest, NullImagesReturnZeroWithoutMemoryRequests) {
  for (uint32_t reg = 4; reg < 16; ++reg)
    wave_->debug_write_sgpr(reg, 0);
  for (bool sample : {false, true}) {
    for (bool d16 : {false, true}) {
      wave_->set_exec(5);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
        for (uint32_t reg = 0; reg < 4; ++reg)
          wave_->debug_write_vgpr(reg, lane, 0xdeadbeef);
      amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
      state.is_load = true;
      ASSERT_TRUE(amdgpu::prepare_image_transfer(*wave_, state, 8, 0, {}, 1, 7, d16, false,
                                                 sample ? 4 : ~0u));
      EXPECT_EQ(state.lane_mask, 0u);
      amdgpu::complete_buffer_format_load(*wave_, *cu_, state);
      for (uint32_t lane : {0u, 2u}) {
        EXPECT_EQ(wave_->debug_read_vgpr(0, lane), 0u);
        EXPECT_EQ(wave_->debug_read_vgpr(1, lane), d16 ? 0xdead0000u : 0u);
        EXPECT_EQ(wave_->debug_read_vgpr(2, lane), d16 ? 0xdeadbeefu : 0u);
      }
      EXPECT_EQ(wave_->debug_read_vgpr(0, 1), 0xdeadbeefu);
      state.is_load = false;
      ASSERT_TRUE(amdgpu::prepare_image_transfer(*wave_, state, 8, 0, {}, 1, 7, d16, false));
      EXPECT_EQ(state.lane_mask, 0u);
    }
  }
  for (uint8_t mask : {uint8_t{3}, uint8_t{15}}) {
    wave_->set_exec(1);
    for (uint32_t reg = 8; reg < 12; ++reg)
      wave_->debug_write_vgpr(reg, 0, 0xdeadbeef);
    std::array<uint32_t, 4> words{};
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4) {
      const auto encoded = rdna4::build_vimage(
          11, {.dim = 1, .dmask = mask, .vdata = 8, .rsrc = 8, .th = 1, .vaddr0 = 2, .vaddr1 = 6});
      std::copy(encoded.begin(), encoded.end(), words.begin());
    } else {
      const auto encoded = rdna3::build_mimg(
          11, {.nsa = 1, .dim = 1, .dmask = mask, .glc = 1, .vaddr = 2, .vdata = 8, .srsrc = 2});
      std::copy(encoded.begin(), encoded.end(), words.begin());
      words[2] = 6;
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->lane_mask, 0u);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    for (uint32_t reg = 0; reg < 4; ++reg)
      EXPECT_EQ(wave_->debug_read_vgpr(8 + reg, 0),
                reg < uint32_t(std::popcount(mask) / 2) ? 0 : 0xdeadbeefu);
  }
}

TEST_P(GraphicsExportTest, HardwareImageLoadReadsTiledUintChannelsAndPacksD16) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 image transfer encoding";
  for (bool a16 : {false, true}) {
    SCOPED_TRACE(a16);
    cu_->l1_vector().flush_all();
    cache_.flush_all();
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    const std::array<uint32_t, 8> descriptor{
        0x1000, (46u << 17) | (3u << 30), 3u << 14, (9u << 28) | (3u << 20) | 0xfac, 0, 0, 0, 0};
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->set_exec(5);
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      wave_->debug_write_vgpr(0, lane, 0xdeadbeef);
      wave_->debug_write_vgpr(1, lane, 0xdeadbeef);
      wave_->debug_write_vgpr(2, lane, (lane == 2 ? 4 : 1) | (a16 ? 2u << 16 : 0));
      wave_->debug_write_vgpr(3, lane, 2);
    }
    // Byte address for texel (1,2) in a 4-byte GFX12 64 KiB tile.
    memory_.write32(0x100000 + 36, 0x44332211);
    const std::array<uint32_t, 4> words{0xd3c00021u | (a16 ? 1u << 6 : 0), 0x00001000,
                                        a16 ? 0x0000ff02u : 0x00000302u, 0};
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction_owner = std::move(decoded).value();
    auto *instruction = instruction_owner.get();
    ASSERT_TRUE(instruction->is_memory_op());
    ASSERT_TRUE(cu_->execute_instruction(instruction, *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    ASSERT_NE(instruction->data(), nullptr);
    EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->wait_counter_type,
              amdgpu::WaitCounterType::LOADCNT);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction_owner.release(), *wave_);
    EXPECT_EQ(wave_->debug_read_vgpr(0, 0), 0x00220011);
    EXPECT_EQ(wave_->debug_read_vgpr(1, 0), 0x00440033);
    EXPECT_EQ(wave_->debug_read_vgpr(0, 1), 0xdeadbeef);
    EXPECT_EQ(wave_->debug_read_vgpr(1, 1), 0xdeadbeef);
    EXPECT_EQ(wave_->debug_read_vgpr(0, 2), 0);
    EXPECT_EQ(wave_->debug_read_vgpr(1, 2), 0);
    wave_->debug_write_vgpr(0, 0, 0x00660055);
    wave_->debug_write_vgpr(1, 0, 0x00880077);
    const auto store = rdna4::build_vimage(6, {.dim = 1,
                                               .d16 = 1,
                                               .a16 = a16,
                                               .dmask = 15,
                                               .rsrc = 8,
                                               .vaddr0 = 2,
                                               .vaddr1 = static_cast<uint8_t>(a16 ? 255 : 3)});
    std::array<uint32_t, 4> store_words{};
    std::copy(store.begin(), store.end(), store_words.begin());
    auto decoded_store = decoder_->decode(store_words.data());
    ASSERT_FALSE(decoded_store.failed());
    auto store_instruction = std::move(decoded_store).value();
    ASSERT_TRUE(store_instruction->is_memory_op());
    ASSERT_TRUE(cu_->execute_instruction(store_instruction.get(), *wave_).succeeded());
    ASSERT_NE(store_instruction->data(), nullptr);
    EXPECT_EQ(store_instruction->data_as<amdgpu::VectorMemState>()->wait_counter_type,
              amdgpu::WaitCounterType::STORECNT);
    pipeline.issue(store_instruction.release(), *wave_);
    cu_->l1_vector().flush_all();
    cache_.flush_all();
    EXPECT_EQ(memory_.read32(0x100000 + 36), 0x88776655);
  }
}

TEST_P(GraphicsExportTest, LinearImageTransfersRespectDefaultPitchAndArrayDescriptorFields) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    const char *name;
    uint32_t width, word4, type, dim, offset;
    uint32_t layer = 0;
    bool out_of_bounds = false;
  };
  const Case cases[] = {
      {"default pitch", 2, 0, 9, 1, gfx12 ? 132u : 260u},
      {"custom pitch", 2, 127, 9, 1, 516},
      {"two array layers", 128, 1, 13, 5, 516},
      {"one array layer", 128, 0, 13, 5, 516},
      {"second array layer", 128, 1, 13, 5, 1540, 1},
      {"nonzero array view", 128, (2 << 16) | 4, 13, 5, 3588, 1},
      {"cube face view", 128, (6 << 16) | 11, 11, 5, 7684, 1},
      {"cube face view limit", 128, (6 << 16) | 11, 11, 5, 516, 6, true},
      {"array view limit", 128, (2 << 16) | 4, 13, 5, 516, 3, true},
      {"array index overflow", 128, (2 << 16) | 4, 13, 5, 516, 0xffffffff, true},
      {"1D resource with array coordinates", 128, 0, 8, 4, 4},
      {"1D second array layer", 128, 1, 12, 4, 516, 1},
      {"1D nonzero array view", 128, (2 << 16) | 4, 12, 4, 1540, 1},
      {"1D array view limit", 128, (2 << 16) | 4, 12, 4, 4, 3, true},
      {"1D array index overflow", 128, (2 << 16) | 4, 12, 4, 4, 0xffffffff, true}};
  for (bool store : {false, true})
    for (bool a16 : {false, true})
      for (uint32_t i = 0; i < std::size(cases); ++i) {
        const auto &c = cases[i];
        SCOPED_TRACE(c.name);
        SCOPED_TRACE(a16);
        SCOPED_TRACE(store);
        const bool one_dimensional = c.dim == 4;
        cu_->l1_vector().flush_all();
        cache_.flush_all();
        cu_->l1_vector().invalidate_all();
        cache_.invalidate_all();
        const uint32_t base = 0x100000 + i * 0x10000;
        const std::array<uint32_t, 8> descriptor{
            base >> 8,
            (46u << (gfx12 ? 17 : 20)) | (((c.width - 1) & 3) << 30),
            ((c.width - 1) >> 2) | (one_dimensional ? 0 : 1u << 14),
            (c.type << 28) | 0xfac,
            c.word4,
            0,
            0,
            0};
        for (uint32_t r = 0; r < descriptor.size(); ++r)
          wave_->debug_write_sgpr(8 + r, descriptor[r]);
        wave_->set_exec(1);
        const uint32_t second_coordinate = one_dimensional ? c.layer : 1;
        wave_->debug_write_vgpr(2, 0, 1 | (a16 ? second_coordinate << 16 : 0));
        wave_->debug_write_vgpr(3, 0, a16 ? (c.layer & 0xffff) | 0xbeef0000 : second_coordinate);
        wave_->debug_write_vgpr(4, 0, c.layer);
        memory_.write32(base + c.offset, 0x44332211);
        for (uint32_t component = 0; component < 4; ++component)
          wave_->debug_write_vgpr(8 + component, 0, 0x21u * (component + 1));
        const uint32_t opcode = store ? 6 : 0; // IMAGE_STORE or IMAGE_LOAD.
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          const auto inst =
              rdna4::build_vimage(opcode, {.dim = uint8_t(c.dim),
                                           .a16 = a16,
                                           .dmask = 15,
                                           .vdata = 8,
                                           .rsrc = 8,
                                           .vaddr0 = 2,
                                           .vaddr1 = 3,
                                           .vaddr2 = static_cast<uint8_t>(a16 ? 255 : 4)});
          std::copy(inst.begin(), inst.end(), words.begin());
        } else {
          const auto inst = rdna3::build_mimg(
              opcode,
              {.dim = uint8_t(c.dim), .dmask = 15, .a16 = a16, .vaddr = 2, .vdata = 8, .srsrc = 2});
          std::copy(inst.begin(), inst.end(), words.begin());
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_NE(instruction->data(), nullptr);
        const auto *transfer = instruction->data_as<amdgpu::VectorMemState>();
        EXPECT_EQ(transfer->lane_mask, c.out_of_bounds ? 0u : 1u);
        if (!c.out_of_bounds) {
          EXPECT_EQ(transfer->per_lane_addr[0], base + c.offset);
        }
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        for (uint32_t component = 0; component < 4; ++component)
          EXPECT_EQ(wave_->debug_read_vgpr(8 + component, 0), store ? 0x21u * (component + 1)
                                                              : c.out_of_bounds
                                                                  ? 0
                                                                  : 0x11u * (component + 1));
        cu_->l1_vector().flush_all();
        cache_.flush_all();
        EXPECT_EQ(memory_.read32(base + c.offset),
                  store && !c.out_of_bounds ? 0x84634221u : 0x44332211u);
      }
}

TEST_P(GraphicsExportTest, CompressedImageTransfersMaterializeClearsAndPreserveStores) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 metadata is allocation-managed";
  constexpr uint64_t base = 0x200000, metadata = 0x100700;
  constexpr uint32_t width = 17, height = 13;
  for (bool a16 : {false, true})
    for (bool array : {false, true}) {
      SCOPED_TRACE(a16);
      SCOPED_TRACE(array);
      cu_->l1_vector().flush_all();
      cache_.flush_all();
      cu_->l1_vector().invalidate_all();
      cache_.invalidate_all();
      // AddrLib: layer 3 of this R_X surface has stride 64KiB and XOR 0x600.
      const uint64_t selected_base = array ? base + 3 * 65536 + 0x600 : base;
      const auto preserved = *amdgpu::gfx11_image_address(base, 11, 10, width, 4, 27);
      memory_.write32(preserved, 0x12345678);
      const std::array<uint32_t, 8> descriptor{
          base >> 8,
          46u << 20,
          ((width - 1) >> 2) | ((height - 1) << 14),
          ((array ? 13u : 9u) << 28) | (27u << 20) | 0xfac,
          array ? (2u << 16) | 4 : 0,
          0,
          (1u << 21) | (1u << 19) | (static_cast<uint32_t>((metadata >> 8) & 255) << 24),
          metadata >> 16};
      for (uint32_t i = 0; i < descriptor.size(); ++i)
        wave_->debug_write_sgpr(8 + i, descriptor[i]);
      wave_->set_exec(5);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        wave_->debug_write_vgpr(2, lane, (lane == 2 ? width : 11) | (a16 ? 10u << 16 : 0));
        wave_->debug_write_vgpr(3, lane, a16 ? 0xbeef0001 : 10);
        wave_->debug_write_vgpr(4, lane, 1);
        for (uint32_t c = 0; c < 4; ++c)
          wave_->debug_write_vgpr(8 + c, lane, 0xdeadbeef);
      }
      const auto tag = *amdgpu::gfx11_metadata_address(metadata, 11, 10, width, height, 4, 27,
                                                       false, true, array ? 3 : 0);
      // Queue the clear tag through L2 to check that the image path observes dirty
      // metadata and does not overwrite it with a stale backing-memory value.
      const uint8_t key = 8;
      memory_.write8(tag, 2);
      cache_.write(tag, &key, 1);
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
      const auto transfer = [&](uint32_t opcode) {
        const auto encoding = rdna3::build_mimg(opcode, {.dim = static_cast<uint8_t>(array ? 5 : 1),
                                                         .dmask = 15,
                                                         .a16 = a16,
                                                         .vaddr = 2,
                                                         .vdata = 8,
                                                         .srsrc = 2});
        std::array<uint32_t, 4> words{};
        std::copy(encoding.begin(), encoding.end(), words.begin());
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        pipeline.issue(instruction.release(), *wave_);
        ASSERT_FALSE(wave_->instruction_execution_failed());
      };
      transfer(0);
      for (uint32_t c = 0; c < 4; ++c) {
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), c == 3 ? 255 : 0);
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 1), 0xdeadbeef);
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 2), 0);
        wave_->debug_write_vgpr(8 + c, 0, 0x11u * (c + 1));
      }
      transfer(6); // IMAGE_STORE
      transfer(0);
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), 0x11u * (c + 1));
      cu_->l1_vector().flush_all();
      cache_.flush_all();
      EXPECT_EQ(memory_.read32(*amdgpu::gfx11_image_address(selected_base, 11, 10, width, 4, 27)),
                0x44332211u);
      EXPECT_EQ(memory_.read32(*amdgpu::gfx11_image_address(selected_base, 10, 10, width, 4, 27)),
                0xff000000u);
      EXPECT_EQ(memory_.read8(tag), 0xff);
      if (array) {
        EXPECT_EQ(memory_.read32(preserved), 0x12345678u);
      }
      // Filtering must also publish dirty texels before materializing metadata.
      const uint32_t updated = 0x66332211;
      cache_.write(*amdgpu::gfx11_image_address(selected_base, 11, 10, width, 4, 27),
                   reinterpret_cast<const uint8_t *>(&updated), 4);
      wave_->debug_write_sgpr(9, 42u << 20);
      wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
      wave_->debug_write_sgpr(5, 0);
      wave_->debug_write_sgpr(6, (1 << 20) | (1 << 22));
      wave_->debug_write_sgpr(7, 0);
      wave_->set_exec(1);
      wave_->debug_write_vgpr(2, 0, std::bit_cast<uint32_t>(11.0f / width));
      wave_->debug_write_vgpr(3, 0, std::bit_cast<uint32_t>(10.5f / height));
      wave_->debug_write_vgpr(4, 0, std::bit_cast<uint32_t>(1.0f));
      // IMAGE_SAMPLE_LZ
      const auto encoding = rdna3::build_mimg(31, {.dim = static_cast<uint8_t>(array ? 5 : 1),
                                                   .dmask = 15,
                                                   .vaddr = 2,
                                                   .vdata = 8,
                                                   .srsrc = 2,
                                                   .ssamp = 1});
      std::array<uint32_t, 4> words{};
      std::copy(encoding.begin(), encoding.end(), words.begin());
      auto decoded = decoder_->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      ASSERT_FALSE(wave_->instruction_execution_failed());
      pipeline.issue(instruction.release(), *wave_);
      const float expected[] = {8.5f / 255, 17.0f / 255, 25.5f / 255, 178.5f / 255};
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), std::bit_cast<uint32_t>(expected[c]));
      // A subsequent dirty metadata clear must affect only the sampled layer.
      cache_.write(tag, &key, 1);
      auto cleared = decoder_->decode(words.data());
      ASSERT_FALSE(cleared.failed());
      instruction = std::move(cleared).value();
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      pipeline.issue(instruction.release(), *wave_);
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), c == 3 ? 0x3f800000u : 0);
      if (array) {
        EXPECT_EQ(memory_.read32(preserved), 0x12345678u);
      }
    }
}

TEST_P(GraphicsExportTest, CompressedMipSamplingKeepsLaneLodsAndArrayLayersSeparate) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 metadata is allocation-managed";
  constexpr uint64_t base = 0x400000, metadata = 0x200000;
  // AddrLib: five 512x512 RGBA8 R_X mips occupy 1441792 bytes per
  // layer, with 65536 metadata bytes. Mip 3 starts the packed tail.
  const std::array<uint32_t, 8> descriptor{base >> 8,
                                           (42u << 20) | (4u << 16) | (3u << 30),
                                           127u | (511u << 14),
                                           (13u << 28) | (27u << 20) | (4u << 16) | 0xfac,
                                           (2u << 16) | 4,
                                           0,
                                           (1u << 21) | (1u << 19),
                                           metadata >> 16};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
  wave_->debug_write_sgpr(5, 4u << 20); // Maximum LOD = 4.
  wave_->debug_write_sgpr(7, 0);
  wave_->set_exec(3);
  for (uint32_t lane = 0; lane < 2; ++lane) {
    wave_->debug_write_vgpr(2, lane, 0);
    wave_->debug_write_vgpr(3, lane, 0);
    wave_->debug_write_vgpr(4, lane, std::bit_cast<uint32_t>(1.0f)); // View-relative layer.
    wave_->debug_write_vgpr(5, lane, std::bit_cast<uint32_t>(2.5f + lane));
  }
  constexpr uint64_t layer_base = base + 3 * 1441792 + 0x600;
  constexpr uint64_t tag2 = metadata + 3 * 65536 + 16384 + 0x600;
  constexpr uint64_t tag3 = metadata + 3 * 65536 + 0x600;
  constexpr uint64_t other_layer_tag = metadata + 2 * 65536 + 0x200;
  for (bool trilinear : {false, true}) {
    memory_.write8(tag2, 8); // Black, opaque.
    memory_.write8(tag3, 2); // White, opaque.
    memory_.write8(other_layer_tag, 0);
    // Mip 4 has no DCC, despite the descriptor's compression-enable bit.
    memory_.write32(*amdgpu::gfx11_image_address(layer_base, 0, 64, 128, 4, 27), 0);
    wave_->debug_write_sgpr(6, (trilinear ? 2u : 1u) << 26);
    const auto encoding = rdna3::build_mimg(
        29, {.dim = 5, .dmask = 15, .vaddr = 2, .vdata = 8, .srsrc = 2, .ssamp = 1});
    std::array<uint32_t, 4> words{};
    std::copy(encoding.begin(), encoding.end(), words.begin());
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    ASSERT_FALSE(wave_->instruction_execution_failed());
    for (uint32_t c = 0; c < 4; ++c) {
      EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), !trilinear || c == 3 ? 0x3f800000u : 0x3f000000u);
      EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 1), trilinear ? 0x3f000000u : 0u);
    }
    EXPECT_EQ(memory_.read8(tag3), 255);
    EXPECT_EQ(memory_.read8(tag2), trilinear ? 255 : 8);
    EXPECT_EQ(memory_.read8(other_layer_tag), 0);
  }
}

TEST_P(GraphicsExportTest, PackedImageTransfersUseRawWordsAndIgnoreDescriptorSelectors) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t base = 0x600000;
  struct Format {
    uint32_t code, bytes;
  };
  // Raw physical captures include sub-DWORD sign extension, noncontiguous
  // DMASKs and the GFX12 replication of logical X in omitted store words.
  for (const auto [format, bytes] :
       {Format{1, 1}, {7, 2}, {14, 2}, {13, 2}, {42, 4}, {48, 8}, {55, 8}, {63, 16}})
    for (uint8_t op : {2, 3, 4, 5, 8, 9})
      for (uint8_t mask : {1, 3, 8, 10, 15}) {
        SCOPED_TRACE(testing::Message() << "format=" << format << " op=" << unsigned(op)
                                        << " mask=" << unsigned(mask));
        const bool store = op >= 8, mip = op == 4 || op == 5 || op == 9;
        const bool signed_load = op == 3 || op == 5;
        std::array<uint32_t, 8> descriptor{
            base >> 8, (format << (gfx12 ? 17 : 20)) | (3u << 30) | (1u << (gfx12 ? 12 : 16)),
            1,         0x80000977,
            0,         0,
            0,         0};
        descriptor[3] |= 1u << (gfx12 ? 15 : 16);
        for (uint32_t i = 0; i < descriptor.size(); ++i)
          wave_->debug_write_sgpr(8 + i, descriptor[i]);
        wave_->set_exec(7);
        constexpr uint32_t input[] = {0xf1f2f3f4, 0xa1a2a3a4, 0xc1c2c3c4, 0xe1e2e3e4};
        for (uint32_t lane = 0; lane < 4; ++lane) {
          wave_->debug_write_vgpr(0, lane, mip || lane != 2 ? 0 : 8);
          wave_->debug_write_vgpr(1, lane, lane);
          for (uint32_t c = 0; c < 4; ++c)
            wave_->debug_write_vgpr(8 + c, lane, store ? input[c] : 0xdeadbeef);
        }
        for (uint32_t i = 0; i < 128; ++i)
          memory_.write32(base + 4 * i, 0x8192a3b4);
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          const auto encoded = rdna4::build_vimage(
              op, {.dim = 0, .dmask = mask, .vdata = 8, .rsrc = 8, .vaddr0 = 0, .vaddr1 = 1});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        } else {
          const auto encoded =
              rdna3::build_mimg(op, {.dim = 0, .dmask = mask, .vaddr = 0, .vdata = 8, .srsrc = 2});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_NE(instruction->amdgpu_memory_issue_info(), nullptr);
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        cu_->l1_vector().flush_all();
        cache_.flush_all();
        if (store) {
          std::array<uint32_t, 4> expected{};
          if (gfx12 && (mask & 1))
            expected.fill(input[0]);
          uint32_t component = 0;
          for (uint32_t c = 0; c < 4; ++c)
            if (mask & (1u << c))
              expected[c] = input[component++];
          for (uint32_t offset = 0; offset < 512; ++offset) {
            const bool written = (offset >= 256 && offset < 256 + bytes) || (mip && offset < bytes);
            const uint32_t word = (offset % 256) / 4;
            const uint8_t value = (written ? expected[word] : 0x8192a3b4) >> (8 * (offset % 4));
            EXPECT_EQ(memory_.read8(base + offset), value) << "offset=" << offset;
          }
        } else {
          for (uint32_t lane = 0; lane < 3; ++lane) {
            uint32_t component = 0;
            for (uint32_t c = 0; c < 4; ++c) {
              if (!(mask & (1u << c)))
                continue;
              uint32_t value = c * 4 < bytes && lane < 2 ? 0x8192a3b4 : 0;
              if (value && bytes < 4) {
                value &= (1u << (bytes * 8)) - 1;
                if (signed_load)
                  value |= ~((1u << (bytes * 8)) - 1);
              }
              EXPECT_EQ(wave_->debug_read_vgpr(8 + component++, lane), value);
            }
            if (component < 4) {
              EXPECT_EQ(wave_->debug_read_vgpr(8 + component, lane), 0xdeadbeefu);
            }
          }
          EXPECT_EQ(wave_->debug_read_vgpr(8, 3), 0xdeadbeefu);
        }
      }
}

TEST_P(GraphicsExportTest, ImageAtomicsUseRawPayloadsAndOptionalReturns) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t base = 0x600000;
  // Physical GFX11/12 results for 0, 1, signed-min, signed-max, unsigned-max,
  // 7, 8 and 9 with operand 8; odd lanes deliberately miss compare-and-swap.
  for (bool wide : {false, true}) {
    const uint64_t max = wide ? ~uint64_t{0} : 0xffffffffu;
    const uint64_t sign = wide ? uint64_t{1} << 63 : uint64_t{1} << 31;
    const std::array<uint64_t, 8> old{0, 1, sign, sign - 1, max, 7, 8, 9};
    const std::array<std::array<uint64_t, 8>, 13> results{{
        {8, 8, 8, 8, 8, 8, 8, 8},
        {8, 1, 8, sign - 1, 8, 7, 8, 9},
        {8, 9, sign + 8, sign + 7, 7, 15, 16, 17},
        {max - 7, max - 6, sign - 8, sign - 9, max - 8, max, 0, 1},
        {0, 1, sign, 8, max, 7, 8, 8},
        {0, 1, 8, 8, 8, 7, 8, 8},
        {8, 8, 8, sign - 1, 8, 8, 8, 9},
        {8, 8, sign, sign - 1, max, 8, 8, 9},
        {0, 0, 0, 8, 8, 0, 8, 8},
        {8, 9, sign + 8, sign - 1, max, 15, 8, 9},
        {8, 9, sign + 8, sign - 9, max - 8, 15, 0, 1},
        {1, 2, 0, 0, 0, 8, 0, 0},
        {8, 0, 8, 8, 8, 6, 7, 8},
    }};
    for (uint32_t op = 10; op <= 22; ++op)
      for (bool returns : {false, true})
        for (bool a16 : {false, true}) {
          SCOPED_TRACE(testing::Message() << "op=" << op << " wide=" << wide
                                          << " returns=" << returns << " a16=" << a16);
          const uint32_t bytes = wide ? 8 : 4;
          const uint32_t data_words = (wide ? 2 : 1) * (op == 11 ? 2 : 1);
          const uint8_t mask = (1u << data_words) - 1;
          const std::array<uint32_t, 8> descriptor{
              base >> 8, ((wide ? 48u : 20u) << (gfx12 ? 17 : 20)) | (3u << 30),
              1,         0x90000fac,
              0,         0,
              0,         0};
          for (uint32_t i = 0; i < descriptor.size(); ++i)
            wave_->debug_write_sgpr(8 + i, descriptor[i]);
          wave_->set_exec(0x1ff); // Lane 8 is OOB; lane 9 is inactive.
          for (uint32_t lane = 0; lane < 10; ++lane) {
            if (wide)
              memory_.write64(base + lane * bytes, old[lane % 8]);
            else
              memory_.write32(base + lane * bytes, old[lane % 8]);
            wave_->debug_write_vgpr(2, lane, lane);
            wave_->debug_write_vgpr(6, lane, 0);
            const uint64_t compare = old[lane % 8] + (lane & 1);
            const std::array<uint32_t, 4> payload{8, wide ? 0u : uint32_t(compare),
                                                  uint32_t(compare), uint32_t(compare >> 32)};
            for (uint32_t i = 0; i < 4; ++i)
              wave_->debug_write_vgpr(8 + i, lane, payload[i]);
          }
          std::array<uint32_t, 4> words{};
          if (gfx12) {
            const auto encoded = rdna4::build_vimage(op, {.dim = 1,
                                                          .a16 = a16,
                                                          .dmask = mask,
                                                          .vdata = 8,
                                                          .rsrc = 8,
                                                          .th = uint8_t(returns),
                                                          .vaddr0 = 2,
                                                          .vaddr1 = 6});
            std::copy(encoded.begin(), encoded.end(), words.begin());
          } else {
            const auto encoded = rdna3::build_mimg(op, {.nsa = 1,
                                                        .dim = 1,
                                                        .dmask = mask,
                                                        .glc = uint8_t(returns),
                                                        .a16 = a16,
                                                        .vaddr = 2,
                                                        .vdata = 8,
                                                        .srsrc = 2});
            std::copy(encoded.begin(), encoded.end(), words.begin());
            words[2] = 6;
          }
          auto decoded = decoder_->decode(words.data());
          ASSERT_FALSE(decoded.failed());
          auto instruction = std::move(decoded).value();
          ASSERT_EQ(instruction->src_operand(0)->size_bits(), int(data_words * 32));
          // The optional result must not kill the comparison operand.
          EXPECT_EQ(instruction->num_dst_operands(), 1 + returns);
          if (returns) {
            EXPECT_EQ(instruction->dst_operand(instruction->num_dst_operands() - 1)->size_bits(),
                      int(bytes * 8));
          }
          const auto *issue = instruction->amdgpu_memory_issue_info();
          ASSERT_NE(issue, nullptr);
          EXPECT_EQ(issue->counter_obligations()[0].wait_counter_type(),
                    returns ? amdgpu::WaitCounterType::LOADCNT : amdgpu::WaitCounterType::STORECNT);
          ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
          ASSERT_FALSE(wave_->instruction_execution_failed());
          ASSERT_NE(instruction->data(), nullptr);
          amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
          pipeline.issue(instruction.release(), *wave_);
          cu_->l1_vector().flush_all();
          cache_.flush_all();
          for (uint32_t lane = 0; lane < 10; ++lane) {
            const uint64_t stored =
                wide ? memory_.read64(base + lane * bytes) : memory_.read32(base + lane * bytes);
            EXPECT_EQ(stored, lane < 8 ? results[op - 10][lane] : old[lane % 8]);
            const uint64_t returned = returns && lane < 9 ? (lane < 8 ? old[lane] : 0) : 8;
            EXPECT_EQ(wave_->debug_read_vgpr(8, lane), uint32_t(returned));
            if (wide) {
              EXPECT_EQ(wave_->debug_read_vgpr(9, lane), uint32_t(returned >> 32));
            }
            if (op == 11) {
              const uint64_t compare = old[lane % 8] + (lane & 1);
              EXPECT_EQ(wave_->debug_read_vgpr(wide ? 10 : 9, lane), uint32_t(compare));
              if (wide) {
                EXPECT_EQ(wave_->debug_read_vgpr(11, lane), uint32_t(compare >> 32));
              }
            }
          }
        }
  }
}

TEST_P(GraphicsExportTest, FloatImageAtomicsMatchPhysicalSpecialValues) {
  if (GetParam() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Floating-point image atomics are GFX12 instructions.";
  constexpr uint32_t base = 0x600000;
  // Raw physical gfx1201 captures, with MODE both zero and 0xff. These
  // operations preserve subnormals and ignore the shader rounding mode.
  constexpr uint32_t old[] = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
                              0x00000001u, 0x80000001u, 0x007fffffu, 0x7f800000u,
                              0xff800000u, 0x7fc12345u, 0x7f812345u, 0x3f808000u,
                              0x00010001u, 0x80018001u, 0x7c007c00u, 0x7e017d01u};
  constexpr uint32_t source[] = {0x80000000u, 0x00000000u, 0x33800000u, 0xbf800000u,
                                 0x00000001u, 0x80000001u, 0x00000001u, 0xff800000u,
                                 0x7f800000u, 0x3f800000u, 0x7fc23456u, 0x3b800000u,
                                 0x00010001u, 0x80018001u, 0xfc00fc00u, 0x7d017e01u};
  constexpr uint32_t expected[5][16] = {
      {0x00000000u, 0x00000000u, 0x3f800000u, 0xc0000000u, 0x00000002u, 0x80000002u, 0x00800000u,
       0xffc00000u, 0xffc00000u, 0x7fc12345u, 0x7fc23456u, 0x3f810000u, 0x00020002u, 0x80030002u,
       0xf8000000u, 0x7e21dc81u},
      {0x80000000u, 0x80000000u, 0x33800000u, 0xbf800000u, 0x00000001u, 0x80000001u, 0x00000001u,
       0xff800000u, 0xff800000u, 0x3f800000u, 0x7fc23456u, 0x3b800000u, 0x00010001u, 0x80018001u,
       0xfc00fc00u, 0x7d017e01u},
      {0x00000000u, 0x00000000u, 0x3f800000u, 0xbf800000u, 0x00000001u, 0x80000001u, 0x007fffffu,
       0x7f800000u, 0x7f800000u, 0x3f800000u, 0x7fc23456u, 0x3f808000u, 0x00010001u, 0x80018001u,
       0x7c007c00u, 0x7e017d01u},
      {0x00000000u, 0x00000000u, 0x40380000u, 0xc3800000u, 0x00000002u, 0x80000002u, 0x007fffffu,
       0xff800000u, 0x7f800000u, 0x7fc12345u, 0x7fc23490u, 0x41a00000u, 0x00020002u, 0x80028002u,
       0xfe00fe00u, 0x7f017e01u},
      {0x00000000u, 0x00000000u, 0x3f800000u, 0xc0000000u, 0x00000002u, 0x80000002u, 0x007fffffu,
       0xffc00000u, 0xffc00000u, 0x7fc12345u, 0x7fc23456u, 0x3f800000u, 0x00020002u, 0x80028002u,
       0x00000000u, 0x7e217e21u}};
  const std::array<uint32_t, 8> descriptor{
      base >> 8, (20u << 17) | (3u << 30), 3, 0x80000fac, 0, 0, 0, 0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  for (uint32_t mode : {0u, 0xffu})
    for (uint32_t op = 131; op <= 135; ++op)
      for (bool returns : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "op=" << op << " mode=" << mode << " returns=" << returns);
        wave_->set_exec(0xffff);
        wave_->set_mode_raw(mode);
        for (uint32_t lane = 0; lane < 16; ++lane) {
          memory_.write32(base + lane * 4, old[lane]);
          wave_->debug_write_vgpr(0, lane, lane);
          wave_->debug_write_vgpr(4, lane, source[lane]);
        }
        const auto words = rdna4::build_vimage(
            op, {.dim = 0, .dmask = 1, .vdata = 4, .rsrc = 8, .th = uint8_t(returns), .vaddr0 = 0});
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        cu_->l1_vector().flush_all();
        cache_.flush_all();
        for (uint32_t lane = 0; lane < 16; ++lane) {
          EXPECT_EQ(memory_.read32(base + lane * 4), expected[op - 131][lane]) << "lane=" << lane;
          EXPECT_EQ(wave_->debug_read_vgpr(4, lane), returns ? old[lane] : source[lane]);
        }
      }
}

TEST_P(GraphicsExportTest, ExplicitMipTransfersUsePerLaneViewRelativeCoordinates) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t base = 0x600000;
  // Physical GFX11/12 captures: 48-wide linear RGBA8 mip offsets are
  // 768, 512, 256, 0. Large unsigned LODs return zero, without wrapping.
  constexpr uint32_t lods[] = {0, 1, 2, 3, 4, 15, 16, 31, 32, 255, 256, 65535, 65536, ~0u};
  for (uint32_t layer = 0; layer < 4; ++layer)
    for (uint32_t i = 0; i < 256; ++i)
      memory_.write32(base + layer * 1024 + i * 4, 0x44000000 | (layer << 16) | i);
  for (uint8_t dim : {0, 1, 4, 5}) {
    const bool array = dim >= 4;
    const uint32_t components = (dim == 0 ? 1 : dim == 5 ? 3 : 2);
    for (uint32_t first = 0; first < 3; ++first)
      for (bool a16 : {false, true})
        for (bool d16 : {false, true}) {
          SCOPED_TRACE(testing::Message() << "dim=" << unsigned(dim) << " first=" << first
                                          << " a16=" << a16 << " d16=" << d16);
          std::array<uint32_t, 8> descriptor{base >> 8,
                                             (46u << (gfx12 ? 17 : 20)) | (3u << 30) |
                                                 (3u << (gfx12 ? 12 : 16)),
                                             11,
                                             ((dim == 0   ? 8u
                                               : dim == 1 ? 9u
                                               : dim == 4 ? 12u
                                                          : 13u)
                                              << 28) |
                                                 0xfac,
                                             array ? (1u << 16) | 2u : 0,
                                             0,
                                             0,
                                             0};
          if (gfx12) {
            descriptor[1] |= first << 25;
            descriptor[3] |= 2u << 15;
          } else {
            descriptor[3] |= (first << 12) | (2u << 16);
          }
          for (uint32_t i = 0; i < descriptor.size(); ++i)
            wave_->debug_write_sgpr(8 + i, descriptor[i]);
          wave_->set_exec((1u << 28) - 1);
          constexpr uint32_t coords[] = {2, 6, 7, 5};
          for (uint32_t lane = 0; lane < 29; ++lane) {
            const bool outside_layer = array && lane == 27;
            std::array<uint32_t, 4> address{lane < 14 || outside_layer ? 0u : 12u};
            if (array)
              address[components - 1] = lane == 27 ? 2 : lane % 2;
            address[components] = outside_layer ? 0 : lods[lane % 14];
            for (uint32_t i = 0; i < 4; ++i)
              wave_->debug_write_vgpr(
                  coords[i], lane,
                  a16 ? i < 2 ? (address[2 * i] & 0xffff) | (address[2 * i + 1] << 16) : 0
                      : address[i]);
            for (uint32_t i = 0; i < 4; ++i)
              wave_->debug_write_vgpr(8 + i, lane, 0xdeadbeef);
          }
          const auto encode = [&](uint32_t opcode) {
            std::array<uint32_t, 4> words{};
            if (gfx12) {
              const auto encoded = rdna4::build_vimage(opcode, {.dim = dim,
                                                                .d16 = d16,
                                                                .a16 = a16,
                                                                .dmask = 15,
                                                                .vdata = 8,
                                                                .rsrc = 8,
                                                                .vaddr0 = 2,
                                                                .vaddr1 = 6,
                                                                .vaddr2 = 7,
                                                                .vaddr3 = 5});
              std::copy(encoded.begin(), encoded.end(), words.begin());
            } else {
              const auto encoded = rdna3::build_mimg(opcode, {.nsa = 1,
                                                              .dim = dim,
                                                              .dmask = 15,
                                                              .a16 = a16,
                                                              .d16 = d16,
                                                              .vaddr = 2,
                                                              .vdata = 8,
                                                              .srsrc = 2});
              std::copy(encoded.begin(), encoded.end(), words.begin());
              words[2] = 6 | (7u << 8) | (5u << 16);
            }
            return words;
          };
          auto words = encode(1); // IMAGE_LOAD_MIP
          auto decoded = decoder_->decode(words.data());
          ASSERT_FALSE(decoded.failed());
          auto instruction = std::move(decoded).value();
          const auto *issue = instruction->amdgpu_memory_issue_info();
          ASSERT_NE(issue, nullptr);
          const auto obligations = issue->counter_obligations();
          ASSERT_EQ(obligations.size(), 1u);
          EXPECT_EQ(obligations[0].wait_counter_type(), amdgpu::WaitCounterType::LOADCNT);
          EXPECT_EQ(obligations[0].completion_class(), amdgpu::MemoryCompletionClass::VMEM);
          ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
          ASSERT_FALSE(wave_->instruction_execution_failed());
          ASSERT_NE(instruction->data(), nullptr);
          amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
          pipeline.issue(instruction.release(), *wave_);
          for (uint32_t lane = 0; lane < 28; ++lane) {
            const bool outside_layer = array && lane == 27;
            const uint32_t lod = outside_layer ? 0 : lods[lane % 14] & (a16 ? 0xffffu : ~0u);
            const uint32_t x = lane < 14 || outside_layer ? 0 : 12;
            const bool valid = lod <= 2 - first && x < (48u >> (first + lod)) && !outside_layer;
            const uint32_t value = valid ? 0x44000000 | ((array ? 1 + lane % 2 : 0) << 16) |
                                               ((3 - first - lod) * 64 + x)
                                         : 0;
            for (uint32_t c = 0; c < (d16 ? 2u : 4u); ++c) {
              const uint32_t expected =
                  d16 ? ((value >> (16 * c)) & 255) | (((value >> (16 * c + 8)) & 255) << 16)
                      : (value >> (8 * c)) & 255;
              EXPECT_EQ(wave_->debug_read_vgpr(8 + c, lane), expected) << "lane=" << lane;
            }
          }
          EXPECT_EQ(wave_->debug_read_vgpr(8, 28), 0xdeadbeefu);
          std::array<uint32_t, 1024> expected{};
          for (uint32_t i = 0; i < expected.size(); ++i)
            expected[i] = 0x44000000 | ((i / 256) << 16) | (i % 256);
          for (uint32_t lane = 0; lane < 29; ++lane) {
            for (uint32_t c = 0; c < (d16 ? 2u : 4u); ++c)
              wave_->debug_write_vgpr(8 + c, lane,
                                      d16 ? (0x11u * (2 * c + 1)) | (0x11u * (2 * c + 2) << 16)
                                          : 0x11u * (c + 1));
            const bool outside_layer = array && lane == 27;
            const uint32_t lod = outside_layer ? 0 : lods[lane % 14] & (a16 ? 0xffffu : ~0u);
            const uint32_t x = lane < 14 || outside_layer ? 0 : 12;
            if (lane < 28 && !outside_layer && lod <= 2 - first && x < (48u >> (first + lod)))
              expected[(array ? 1 + lane % 2 : 0) * 256 + (3 - first - lod) * 64 + x] = 0x44332211;
          }
          words = encode(7); // IMAGE_STORE_MIP
          decoded = decoder_->decode(words.data());
          ASSERT_FALSE(decoded.failed());
          instruction = std::move(decoded).value();
          issue = instruction->amdgpu_memory_issue_info();
          ASSERT_NE(issue, nullptr);
          ASSERT_EQ(issue->counter_obligations().size(), gfx12 ? 1u : 2u);
          if (!gfx12) {
            EXPECT_EQ(issue->counter_obligations()[1].wait_counter_type(),
                      amdgpu::WaitCounterType::EXPCNT);
          }
          EXPECT_EQ(issue->counter_obligations()[0].wait_counter_type(),
                    amdgpu::WaitCounterType::STORECNT);
          ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
          ASSERT_FALSE(wave_->instruction_execution_failed());
          pipeline.issue(instruction.release(), *wave_);
          cu_->l1_vector().flush_all();
          cache_.flush_all();
          for (uint32_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(memory_.read32(base + 4 * i), expected[i]) << "word=" << i;
            memory_.write32(base + 4 * i, 0x44000000 | ((i / 256) << 16) | (i % 256));
          }
        }
  }
}

TEST_P(GraphicsExportTest, LinearMipLevelsUseReverseAllocationOrder) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  // The odd-width offsets were also checked by mapping physical RDNA3/RDNA4
  // image backing after clears; a load/store round trip alone can hide them.
  struct Case {
    uint32_t width;
    std::array<uint32_t, 4> offsets;
  };
  const Case cases[] = {{48, {1024, 512, 256, 0}}, {129, {2560, 1280, 512, 0}}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.width);
    const uint32_t base = 0x300000 + test.width * 0x1000;
    for (uint32_t level = 0; level < 4; ++level) {
      SCOPED_TRACE(level);
      std::array<uint32_t, 8> descriptor{base >> 8, 0, 11, (8u << 28) | 0xfac, 0, 0, 0, 0};
      descriptor[1] =
          (63u << (gfx12 ? 17 : 20)) | (((test.width - 1) & 3u) << 30) | (3u << (gfx12 ? 12 : 16));
      descriptor[2] = (test.width - 1) >> 2;
      if (gfx12) {
        descriptor[1] |= level << 25;
        descriptor[3] |= 3u << 15;
      } else {
        descriptor[3] |= (level << 12) | (3u << 16);
      }
      for (uint32_t r = 0; r < descriptor.size(); ++r)
        wave_->debug_write_sgpr(8 + r, descriptor[r]);
      wave_->set_exec(1);
      wave_->debug_write_vgpr(2, 0, 5);
      const uint32_t address = base + test.offsets[level] + 5 * 16;
      for (uint32_t c = 0; c < 4; ++c)
        memory_.write32(address + 4 * c, std::bit_cast<uint32_t>(float(level + c)));
      std::array<uint32_t, 4> words{};
      if (gfx12) {
        const auto inst =
            rdna4::build_vimage(0, {.dim = 0, .dmask = 15, .vdata = 8, .rsrc = 8, .vaddr0 = 2});
        std::copy(inst.begin(), inst.end(), words.begin());
      } else {
        const auto inst =
            rdna3::build_mimg(0, {.dim = 0, .dmask = 15, .vaddr = 2, .vdata = 8, .srsrc = 2});
        std::copy(inst.begin(), inst.end(), words.begin());
      }
      auto decoded = decoder_->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      ASSERT_NE(instruction->data(), nullptr);
      EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->per_lane_addr[0], address);
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
      pipeline.issue(instruction.release(), *wave_);
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), std::bit_cast<uint32_t>(float(level + c)));
    }
  }
}

// AddrLib offsets for the last texel of all eight 200x180 RGBA8 mip levels.
constexpr uint32_t kMipLastTexelOffsets[2][8] = {
    // GFX11
    {333692, 121628, 52868, 12680, 26676, 1220, 536, 8960},
    // GFX12
    {365692, 124188, 47492, 20104, 9012, 4292, 2072, 1536},
};

TEST_P(GraphicsExportTest, ImageStoreMasksFillOmittedChannelsAtBothDataWidths) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool d16 : {false, true})
    for (uint32_t mask = 1; mask < 16; ++mask) {
      SCOPED_TRACE(testing::Message() << "d16=" << d16 << " mask=" << mask);
      const std::array<uint32_t, 8> descriptor{
          0x1000, 46u << (gfx12 ? 17 : 20), 0, 0x90000fac, 0, 0, 0, 0};
      for (uint32_t i = 0; i < descriptor.size(); ++i)
        wave_->debug_write_sgpr(8 + i, descriptor[i]);
      wave_->set_exec(1);
      wave_->debug_write_vgpr(0, 0, 0);
      wave_->debug_write_vgpr(1, 0, 0);
      for (uint32_t i = 0; i < 4; ++i)
        wave_->debug_write_vgpr(
            4 + i, 0, d16 ? 0x11u * (2 * i + 1) | (0x11u * (2 * i + 2) << 16) : 0x11u * (i + 1));
      amdgpu::VectorMemState data(amdgpu::GLOBAL_MEM);
      data.is_load = false;
      ASSERT_TRUE(amdgpu::prepare_image_transfer(*wave_, data, 8, 4, {0, 1}, 1, mask, d16, false));
      ASSERT_EQ(data.store_data.size(), wave_->wf_size() * 4);
      const uint32_t channel_mask = mask;
      uint8_t next = 0x11;
      for (uint32_t i = 0; i < 4; ++i) {
        const uint8_t expected = channel_mask & (1u << i) ? std::exchange(next, next + 0x11)
                                 : gfx12                  ? 0x11
                                                          : 0;
        EXPECT_EQ(data.store_data[i], expected);
      }
    }
}

TEST_P(GraphicsExportTest, VolumeTransfersUseDepthBoundsAndIgnoreSrvBaseArray) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (20u << (gfx12 ? 17 : 20)),
                                           32 | (70u << 14),
                                           (10u << 28) | ((gfx12 ? 7u : 29u) << 20) | 0x204,
                                           (19u << 16) | 36,
                                           0,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->set_exec(3);
  for (uint32_t dim : {2u, 5u})
    for (bool a16 : {false, true}) {
      for (uint32_t lane = 0; lane < 2; ++lane) {
        wave_->debug_write_vgpr(0, lane, 21 | (a16 ? 59u << 16 : 0));
        wave_->debug_write_vgpr(1, lane, a16 ? (lane ? 37 : 12) : 59);
        wave_->debug_write_vgpr(2, lane, lane ? 37 : 12);
        wave_->debug_write_vgpr(4, lane, 0x12345678);
      }
      for (bool load : {false, true}) {
        amdgpu::VectorMemState data(amdgpu::GLOBAL_MEM);
        data.is_load = load;
        ASSERT_TRUE(amdgpu::prepare_image_transfer(*wave_, data, 8, 4, {0, 1, 2}, dim, 1, false,
                                                   false, ~0u, amdgpu::ImageSampleMode::Implicit,
                                                   a16));
        EXPECT_EQ(data.lane_mask, 1u);
        // 256KB 3D AddrLib address for (21,59,12), without mipmaps.
        EXPECT_EQ(data.per_lane_addr[0], 0x100000u + (gfx12 ? 192076u : 189004u));
        if (!load) {
          ASSERT_GE(data.store_data.size(), 4u);
          uint32_t value;
          std::memcpy(&value, data.store_data.data(), 4);
          EXPECT_EQ(value, 0x12345678u);
        }
      }
    }
}

TEST_P(GraphicsExportTest, TiledMipTransfersUseViewBoundsAndIndependentBackingOffsets) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t base = 0x400000;
  for (uint8_t dim : {1, 2, 5})
    for (bool compressed : {false, true}) {
      const bool volume = dim != 1;
      if (gfx12 && compressed)
        continue;
      for (bool explicit_mip : {false, true})
        for (uint32_t level = 0; level < 8; ++level) {
          SCOPED_TRACE(testing::Message() << "level=" << level << " explicit=" << explicit_mip
                                          << " dim=" << unsigned(dim));
          const uint32_t width = std::max(1u, 200u >> level), height = std::max(1u, 180u >> level);
          const uint32_t layer = volume && level < 7 ? 1 : 0;
          const auto pixels =
              amdgpu::image_mip_layout(gfx12, gfx12 ? 3 : 27, 4, 200, 180, 8, level);
          ASSERT_TRUE(pixels);
          std::array<uint32_t, 8> descriptor{
              base >> 8,
              (46u << (gfx12 ? 17 : 20)) | (3u << 30) | (7u << (gfx12 ? 12 : 16)),
              49u | (179u << 14),
              ((volume ? 10u : 9u) << 28) | ((gfx12 ? 3u : 27u) << 20) | 0xfac,
              0,
              0,
              0,
              0};
          if (volume)
            descriptor[4] = 127;
          if (gfx12) {
            descriptor[1] |= level << 25;
            descriptor[3] |= 7u << 15;
          } else {
            descriptor[3] |= (level << 12) | (7u << 16);
          }
          constexpr uint64_t metadata = 0x200000;
          // AddrLib: 200x180 R_X has three DCC blocks; tail levels after mip 2
          // cannot use metadata. Seed the first byte of every block independently.
          if (compressed) {
            descriptor[6] = (1u << 21) | (1u << 19);
            descriptor[7] = metadata >> 16;
          }
          const uint64_t tag = *amdgpu::gfx11_metadata_address(
              (metadata + (2 - std::min(level, 2u)) * 16384 + layer * 49152) ^
                  (amdgpu::gfx11_image_slice_xor(layer, 4, 27) & 16383),
              width - 1, height - 1, (200 + (1u << level) - 1) >> level,
              (180 + (1u << level) - 1) >> level, 4, 27, false);
          memory_.write8(tag, 2);
          for (uint32_t r = 0; r < descriptor.size(); ++r)
            wave_->debug_write_sgpr(8 + r, descriptor[r]);
          wave_->set_exec(15);
          for (uint32_t lane = 0; lane < 4; ++lane) {
            wave_->debug_write_vgpr(2, lane, lane == 1 ? width : lane == 3 ? ~0u : width - 1);
            wave_->debug_write_vgpr(3, lane, lane == 2 ? height : height - 1);
            wave_->debug_write_vgpr(4, lane, volume ? layer : level);
            wave_->debug_write_vgpr(5, lane, level);
          }
          const uint64_t address =
              (base + kMipLastTexelOffsets[gfx12][level] + layer * pixels->slice_size) ^
              (gfx12 ? 0 : amdgpu::gfx11_image_slice_xor(layer, 4, 27));
          memory_.write32(address, 0x44332211);
          amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
          for (bool store : {false, true}) {
            // The load has expanded this block. Exercise a store directly into a
            // different compressed clear, including the first packed-tail mip.
            if (store && compressed && level <= 2)
              memory_.write8(tag, 8);
            const uint32_t first = explicit_mip ? 0 : level;
            wave_->debug_write_sgpr(gfx12 ? 9 : 11,
                                    gfx12 ? (descriptor[1] & ~(31u << 25)) | (first << 25)
                                          : (descriptor[3] & ~(15u << 12)) | (first << 12));
            const uint8_t opcode = (store ? 6 : 0) + explicit_mip;
            std::array<uint32_t, 4> words{};
            if (gfx12) {
              const auto inst = rdna4::build_vimage(opcode, {.dim = dim,
                                                             .dmask = 15,
                                                             .vdata = 8,
                                                             .rsrc = 8,
                                                             .vaddr0 = 2,
                                                             .vaddr1 = 3,
                                                             .vaddr2 = 4,
                                                             .vaddr3 = 5});
              std::copy(inst.begin(), inst.end(), words.begin());
            } else {
              const auto inst = rdna3::build_mimg(
                  opcode, {.dim = dim, .dmask = 15, .vaddr = 2, .vdata = 8, .srsrc = 2});
              std::copy(inst.begin(), inst.end(), words.begin());
            }
            auto decoded = decoder_->decode(words.data());
            ASSERT_FALSE(decoded.failed());
            auto instruction = std::move(decoded).value();
            ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
            ASSERT_FALSE(wave_->instruction_execution_failed());
            ASSERT_NE(instruction->data(), nullptr);
            const auto *transfer = instruction->data_as<amdgpu::VectorMemState>();
            EXPECT_EQ(transfer->per_lane_addr[0], address);
            EXPECT_EQ(transfer->lane_mask, 1u);
            pipeline.issue(instruction.release(), *wave_);
            if (!store) {
              for (uint32_t c = 0; c < 4; ++c) {
                EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0),
                          compressed && level <= 2 ? 255u : 0x11u * (c + 1));
                for (uint32_t lane = 1; lane < 4; ++lane)
                  EXPECT_EQ(wave_->debug_read_vgpr(8 + c, lane), 0u);
                wave_->debug_write_vgpr(8 + c, 0, 0x80u + c);
              }
            }
          }
          cu_->l1_vector().flush_all();
          cache_.flush_all();
          EXPECT_EQ(memory_.read32(address), 0x83828180u);
          if (compressed && level <= 2) {
            const auto mip = amdgpu::image_mip_layout(false, 27, 4, 200, 180, 8, level);
            ASSERT_TRUE(mip);
            const auto neighbor = amdgpu::gfx11_image_address(
                amdgpu::image_layer_base(false, base + mip->offset, mip->slice_size, layer, 4, 27),
                width - 2 + mip->tail_x, height - 1 + mip->tail_y, mip->pitch, 4, 27);
            ASSERT_TRUE(neighbor);
            EXPECT_EQ(memory_.read32(*neighbor), 0xff000000u);
          }
          EXPECT_EQ(memory_.read8(tag), compressed && level <= 2 ? 255 : 2);
        }
    }
}

TEST_P(GraphicsExportTest, NormalizedBlendingUsesHardwarePrecisionAndQuadBypass) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    const char *name;
    bool srgb;
    uint32_t epsilon, blend, opt;
    std::array<uint32_t, 4> initial, expected;
    std::array<std::array<uint32_t, 2>, 4> exported;
    std::array<uint32_t, 4> constants{0xbf400000, 0x3e99999a, 0x3fa00000, 0x3f266666};
    bool triangle = false;
    uint32_t exported_lanes = 15;
  };
  // Raw physical RDNA3/4 quad captures. Epsilon is programmed state: run both
  // settings on each simulated architecture, including RDNA3.5.
  const Case cases[] = {
      // Physical RDNA3/4 round the blend accumulator before the sRGB encoder
      // truncates to FP16. Without that rounding, each RGB channel is one low.
      {"srgb accumulator rounding",
       true,
       6,
       0x60010501u,
       0,
       {0x6fb0b0b0u, 0x76b5b5b5u, 0x3b848484u, 0x357e7e7eu},
       {0x09cfcfcfu, 0x02d4d4d4u, 0x2ad0d0d0u, 0x3bd2d2d2u},
       {{{0x32753275u, 0x285b3275u},
         {0x32433243u, 0x1fd83243u},
         {0x36f436f4u, 0x315136f4u},
         {0x37b037b0u, 0x336f37b0u}}}},
      // Physical RDNA3/4 include an uncovered helper's export in the quad
      // decision. Without that export, the near-one alpha bypasses blending.
      {.name = "uncovered helper vetoes source copy",
       .srgb = false,
       .epsilon = 6,
       .blend = 0x40000504u,
       .opt = 0x01540154u,
       .initial = {0x01bdc1bdu, 0x01bdc1bdu, 0x01bdc1bdu, 0x01bdc1bdu},
       .expected = {0x01bdc1bdu, 0x01bdc1bdu, 0xfe000000u, 0x01bdc1bdu},
       .exported = {{{0, 0}, {0, 0}, {0, 0x3bfd0000u}, {0, 0x3b750000u}}},
       .triangle = true},
      {.name = "unexported helper permits source copy",
       .srgb = false,
       .epsilon = 6,
       .blend = 0x40000504u,
       .opt = 0x01540154u,
       .initial = {0x01bdc1bdu, 0x01bdc1bdu, 0x01bdc1bdu, 0x01bdc1bdu},
       .expected = {0x01bdc1bdu, 0x01bdc1bdu, 0xff000000u, 0x01bdc1bdu},
       .exported = {{{0, 0}, {0, 0}, {0, 0x3bfd0000u}, {0, 0x3b750000u}}},
       .triangle = true,
       .exported_lanes = 7},
      {"srgb add exact",
       true,
       0,
       0x61010101u,
       0x01110111u,
       {0x00000000u, 0x972553c5u, 0x01080220u, 0x962d51e5u},
       {0x00030000u, 0x982d53c5u, 0x01ff0220u, 0x96ff51e5u},
       {{{0x00b30000u, 0x0151132bu},
         {0x055c0001u, 0x1e061f6cu},
         {0xa9b30100u, 0xb651542bu},
         {0xae5c0101u, 0xd306606cu}}}},
      {"srgb add epsilon",
       true,
       6,
       0x61010101u,
       0x01110111u,
       {0x00000000u, 0x972553c5u, 0x01080220u, 0x962d51e5u},
       {0x00000000u, 0x982d53c5u, 0x01ff0220u, 0x96ff51e5u},
       {{{0x00b30000u, 0x0151132bu},
         {0x055c0001u, 0x1e061f6cu},
         {0xa9b30100u, 0xb651542bu},
         {0xae5c0101u, 0xd306606cu}}}},
      {"srgb NaN flags",
       true,
       6,
       0x61010101u,
       0x01110111u,
       {0xd646ef91u, 0x6d634054u, 0xd74eedb1u, 0x6c6b4274u},
       {0xffffef91u, 0xff634054u, 0xd74effb1u, 0x6d6b4274u},
       {{{0xce5d009au, 0x46337245u},
         {0xd306009bu, 0x62e87e00u},
         {0x775d019au, 0xfb33b345u},
         {0x7e00019bu, 0x17e8bf86u}}}},
      {"srgb source quad",
       true,
       6,
       0x60000000u,
       0x01000100u,
       {0xf210fbcdu, 0x893d4c80u, 0xf318f9edu, 0x88354ea0u},
       {0x00000100u, 0x00000200u, 0x00000000u, 0x00000000u},
       {{{0x0b4100deu, 0xe647b389u},
         {0x0fea00dfu, 0x02fcbfcau},
         {0xb44101deu, 0x9b47f489u},
         {0xb8ea01dfu, 0xb7fc00cau}}}},
      {"srgb source quad veto",
       true,
       6,
       0x60000000u,
       0x01000100u,
       {0x00000000u, 0x972553c5u, 0x01080220u, 0x962d51e5u},
       {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       {{{0x00b30000u, 0x0151132bu},
         {0x055c0001u, 0x1e061f6cu},
         {0xa9b30100u, 0xb651542bu},
         {0xae5c0101u, 0xd306606cu}}}},
      {"srgb retired quad pixels",
       true,
       6,
       0x65040504u,
       0x01540154u,
       {0xc2dff8a3u, 0x5d224bfeu, 0xc3d7fa83u, 0x5c2a49deu},
       {0xc2dff8a3u, 0x5d224bfeu, 0xff000000u, 0xff000000u},
       {{{0x188fe67cu, 0x86fd68a7u},
         {0x1d38e67du, 0xa3b274e8u},
         {0xc18fe77cu, 0x3bfda9a7u},
         {0xc638e77du, 0x58b2b5e8u}}}},
      {"srgb neighboring arithmetic",
       true,
       6,
       0x65040504u,
       0x01540154u,
       {0xf0b8787bu, 0x675dab3eu, 0xf1b07a5bu, 0x6655a91eu},
       {0xff020000u, 0x675dab3eu, 0xefb17a5bu, 0xffff0100u},
       {{{0x088bd818u, 0x6a491143u},
         {0x0d34d819u, 0x86fe1d84u},
         {0xb18bd918u, 0x1f495243u},
         {0xb634d919u, 0x3bfe5e84u}}}},
      {"srgb alpha precision",
       true,
       6,
       0x63020302u,
       0x01540132u,
       {0xf67f9023u, 0x6982237eu, 0xf7779203u, 0x688a215eu},
       {0xffff9023u, 0x69ff237eu, 0xf7779203u, 0x5f8a215eu},
       {{{0xe48fd27cu, 0x62fd54a7u},
         {0xe938d27du, 0x7e0060e8u},
         {0x8d8fd37cu, 0x17fd95a7u},
         {0x9238d37du, 0x34b2a1e8u}}}},
      {"srgb blend constants",
       true,
       0,
       0x6c0b0c0bu,
       0x01770177u,
       {0x0cdbf581u, 0xf386a67cu, 0x0dd3f7a1u, 0xf28ea45cu},
       {0x0400f881u, 0x6200c67cu, 0x0500d3a1u, 0x55008b5cu},
       {{{0x447f38ecu, 0x102d9717u},
         {0x492838edu, 0x2ce2a358u},
         {0xed7f39ecu, 0xc52dd817u},
         {0xf22839edu, 0xe1e2e458u}}}},
      {"srgb reverse subtract",
       true,
       6,
       0x61810181u,
       0x05110511u,
       {0x80d0cfdfu, 0x1bfd1812u, 0x81d8cdffu, 0x1af51a32u},
       {0x8000cfdfu, 0x1bfd1512u, 0x00d8cdffu, 0x00f51a32u},
       {{{0x12b102ceu, 0x84f77179u},
         {0x175a02cfu, 0xa1ac7e00u},
         {0xbbb103ceu, 0x39f7b279u},
         {0xc05a03cfu, 0x56acbebau}}}},
      {"unorm source NaNs",
       false,
       6,
       0x61010101u,
       0x01110111u,
       {0x74f2a158u, 0xe3b63a1au, 0x75e0a485u, 0xe2a83832u},
       {0x74ffa158u, 0xe3ff3a1au, 0x75e0a485u, 0xe2a83832u},
       {{{0xe1ab7e00u, 0xf4e93d63u},
         {0xe6547e00u, 0x119e49a4u},
         {0x8aab7e00u, 0xa9e97e00u},
         {0x8f547e00u, 0xc69e8aa4u}}}},
      {"unorm alpha NaNs",
       false,
       6,
       0x65040504u,
       0x01540154u,
       {0x74f2a158u, 0xe3b63a1au, 0x75e0a485u, 0xe2a83832u},
       {0x74f2a158u, 0xe3b63a1au, 0x75e0a485u, 0xe2a83832u},
       {{{0xe1ab7e00u, 0xf4e93d63u},
         {0xe6547e00u, 0x119e49a4u},
         {0x8aab7e00u, 0xa9e97e00u},
         {0x8f547e00u, 0xc69e8aa4u}}}},
      {"unorm epsilon",
       false,
       6,
       0x61010101u,
       0x01110111u,
       {0xec09400eu, 0x9b010b01u, 0xed0c421fu, 0x9a020a05u},
       {0xec09410fu, 0x9b030d02u, 0xedff4220u, 0x9aff0a06u},
       {{{0x1bc31c90u, 0xf32113bbu},
         {0x206c1c91u, 0x0fd61ffcu},
         {0xc4c31d90u, 0xa82154bbu},
         {0xc96c1d91u, 0xc4d660fcu}}}},
      {"unorm constant precision",
       false,
       6,
       0x6c0b0c0bu,
       0x01770177u,
       {0x0cb5e938u, 0xf33d6133u, 0x0da6ed5bu, 0xf2455f1bu},
       {0x0400f038u, 0x62009033u, 0x0500a65bu, 0x5500431bu},
       {{{0x447f38ecu, 0x102d9717u},
         {0x492838edu, 0x2ce2a358u},
         {0xed7f39ecu, 0xc52dd817u},
         {0xf22839edu, 0xe1e2e458u}}}},
      {"srgb NaN color constants",
       true,
       0,
       0x6c0b0c0bu,
       0x01770177u,
       {0xdcbeb475u, 0x6363e748u, 0xddb6b655u, 0x626be568u},
       {0xdc00b475u, 0x6300e748u, 0xdd00b655u, 0x6201e568u},
       {{{0xaadf084cu, 0x2f0dbe77u},
         {0xaf88084du, 0x4bc2cab8u},
         {0x53df094cu, 0xe40dfe00u},
         {0x5888094du, 0x00c20bb8u}}},
       {0x7fc00001u, 0x80000000u, 0x7f800000u, 0xff800000u}},
      {"srgb NaN alpha constants",
       true,
       0,
       0x72111211u,
       0x01770177u,
       {0xdcbeb475u, 0x6363e748u, 0xddb6b655u, 0x626be568u},
       {0xdcbeb475u, 0x6363e748u, 0xddb6b655u, 0x626be568u},
       {{{0xaadf084cu, 0x2f0dbe77u},
         {0xaf88084du, 0x4bc2cab8u},
         {0x53df094cu, 0xe40dfe00u},
         {0x5888094du, 0x00c20bb8u}}},
       {0x3e800000u, 0x3f000000u, 0x3f400000u, 0xffc00001u}},
      {"unorm NaN color constants",
       false,
       6,
       0x6c0b0c0bu,
       0x01770177u,
       {0xdc83742du, 0x6320cc11u, 0xdd777717u, 0x6225c823u},
       {0xdc00742du, 0x6300cc11u, 0xdd007717u, 0x6200c823u},
       {{{0xaadf084cu, 0x2f0dbe77u},
         {0xaf88084du, 0x4bc2cab8u},
         {0x53df094cu, 0xe40dfe00u},
         {0x5888094du, 0x00c20bb8u}}},
       {0x7fc00001u, 0x80000000u, 0x7f800000u, 0xff800000u}},
      {"unorm NaN alpha constants",
       false,
       6,
       0x72111211u,
       0x01770177u,
       {0xdc83742du, 0x6320cc11u, 0xdd777717u, 0x6225c823u},
       {0xdc83742du, 0x6320cc11u, 0xdd777717u, 0x6225c823u},
       {{{0xaadf084cu, 0x2f0dbe77u},
         {0xaf88084du, 0x4bc2cab8u},
         {0x53df094cu, 0xe40dfe00u},
         {0x5888094du, 0x00c20bb8u}}},
       {0x3e800000u, 0x3f000000u, 0x3f400000u, 0xffc00001u}},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    auto state = rectangle_state();
    auto &ctx = state.context_registers;
    ctx[gfx12 ? 0x3b0 : 0x31c] = 10 | ((test.srgb ? 6u : 0u) << 8);
    ctx[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
    ctx[gfx12 ? 0x31f : 0x3b8] = gfx12 ? 3u << 15 : 26u << 14;
    ctx[0x318] = 0x1000;
    ctx[gfx12 ? 0x214 : 0x8e] = ctx[gfx12 ? 0x215 : 0x8f] = 15;
    ctx[gfx12 ? 0x195 : 0x1c5] = 4; // FP16 ABGR export.
    ctx[gfx12 ? 0x216 : 0x202] = 0xcc0010;
    ctx[0x1e0] = test.blend;
    ctx[0x1d5] = 5; // 8-bit normalized downconvert.
    ctx[0x1d6] = test.epsilon;
    ctx[0x1d8] = test.opt;
    for (uint32_t c = 0; c < 4; ++c)
      ctx[0x105 + c] = test.constants[c];
    ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(2.0f);
    if (test.triangle) {
      // The triangle covers x+y < 2.5, leaving only lane 3 as a helper.
      state.uconfig_registers[0x242] = 4;
      ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(1.25f);
    }
    ctx[0x91] = gfx12 ? 1 | (1 << 16) : 2 | (2 << 16);
    const auto address = [&](uint32_t pixel) {
      return gfx12 ? *amdgpu::gfx12_image_address(0x100000, pixel & 1, pixel >> 1, 4, 4, 3)
                   : *amdgpu::gfx11_image_address(0x100000, pixel & 1, pixel >> 1, 4, 4, 26);
    };
    for (uint32_t pixel = 0; pixel < 4; ++pixel)
      memory_.write32(address(pixel), test.initial[pixel]);
    auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
    export_rectangle_vertices(*draw);
    ASSERT_TRUE(draw->advance(*access_));
    initialize_fragment(draw);
    for (uint32_t pixel = 0; pixel < 4; ++pixel) {
      EXPECT_EQ(amdgpu::GraphicsDrawTestAccess::covered(*draw, 0, pixel),
                !test.triangle || pixel != 3);
      if (test.exported_lanes & (1u << pixel))
        draw->export_lane(*wave_, pixel, 0, 3,
                          {test.exported[pixel][0], test.exported[pixel][1], 0, 0});
    }
    EXPECT_FALSE(draw->advance(*access_));
    for (uint32_t pixel = 0; pixel < 4; ++pixel)
      EXPECT_EQ(memory_.read32(address(pixel)), test.expected[pixel]) << pixel;
  }
}

TEST_P(GraphicsExportTest, ColorBlendingPreservesMasksAndUsesSeparateAlpha) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    uint32_t blend, mask, expected;
    bool reject = false;
    bool constant_boundary = false;
    uint32_t number_format = 0, export_format = 9;
    uint32_t initial = 0xff40bf80;
    uint32_t color_control = 0xcc0010;
    std::array<uint32_t, 4> exported{0x3f000000, 0x3e800000, 0x3e800000, 0x3f000000};
    std::array<float, 4> constants{0.25f, 0.25f, 0.5f, 0.5f};
  };
  constexpr uint32_t enabled = 1u << 30, separate = 1u << 29;
  const Case cases[] = {
      // Physical RDNA3/4 round the aligned UNORM blend sum to FP32. These
      // premultiplied-over results straddle three different quantization ties.
      {.blend = enabled | 1 | (5 << 8),
       .mask = 15,
       .expected = 0x8b6d7679,
       .export_format = 4,
       .initial = 0x6850575a,
       .exported = {0x326f3289, 0x338631f0, 0, 0}},
      {.blend = enabled | 1 | (5 << 8),
       .mask = 15,
       .expected = 0x52414341,
       .export_format = 4,
       .initial = 0x4e3d3f3e,
       .exported = {0x24f824bc, 0x259b24dc, 0, 0}},
      {.blend = enabled | 1 | (5 << 8),
       .mask = 15,
       .expected = 0x6a545b5d,
       .export_format = 4,
       .initial = 0x45353837,
       .exported = {0x31cc3217, 0x326b312d, 0, 0}},
      // Physical RDNA3/4 give blending precedence over XOR, including ZERO/ZERO.
      {.blend = enabled, .mask = 15, .expected = 0, .color_control = 0x660010},
      {.blend = enabled | 1, .mask = 15, .expected = 0x80404080, .color_control = 0x660010},
      {0, 5, 0xff40bf80},
      {enabled | 4 | (5 << 8), 15, 0xbf407f80},
      {enabled | separate | 4 | (5 << 8) | (1 << 16), 15, 0x80407f80},
      {enabled | (2 << 5), 15, 0x80404080},
      {enabled | (3 << 5), 15, 0xff40bf80},
      {enabled | 11, 15, 0x40201020},
      {enabled | 10, 15, 0x80000000},
      {.blend = enabled | 13, .mask = 15, .expected = 0, .reject = true},
      {.blend = enabled | (5 << 5), .mask = 15, .expected = 0, .reject = true},
      {.blend = enabled | 1, .mask = 15, .expected = 0, .reject = true, .number_format = 4},
      // FP32 sRGB exports remain unsupported.
      {.blend = 0, .mask = 15, .expected = 0, .reject = true, .number_format = 6},
      {.blend = enabled | 11, .mask = 5, .expected = 0xff20bf20},
      {.blend = enabled | separate | 11 | (17 << 16),
       .mask = 15,
       .expected = 0x9f6a3500,
       .constant_boundary = true,
       .exported = {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000}},
      // Physical RDNA3/RDNA4 witnesses distinguish twelve-bit destination
      // normalization from both full FP32 and FP16 precision.
      {.blend = enabled | separate | 4 | (5 << 8) | (1 << 16) | (1 << 24),
       .mask = 15,
       .expected = 0xc6c999a8,
       .initial = 0x10c60035,
       .exported = {0x3f57c000, 0x3f57c000, 0x3f4ac000, 0x3f36a000}},
      {.blend = enabled | separate | 4 | (5 << 8) | (1 << 16) | (1 << 24),
       .mask = 15,
       .expected = 0x8d5c155a,
       .initial = 0x20760005,
       .exported = {0x3f4bc000, 0x3e40c000, 0x3e60e000, 0x3edac000}},
      // Physical HPP depth-peeling inputs: alignment's sticky bit breaks the
      // FP32 tie upward. Dropping it gives 141 instead of 142.
      {.blend = enabled | separate | 4 | (3 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0xef02968e,
       .initial = 0x01020135,
       .exported = {0x3ef36000, 0x3f1fa000, 0, 0x3f702000}},
      // Independent physical readbacks distinguish 23-bit product alignment
      // and inverse-factor decomposition from exact host-double arithmetic.
      {.blend = enabled | separate | 2 | (3 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0x68a2a2a2,
       .initial = 0x25252525,
       .exported = {0x3f472000, 0x3f472000, 0x3f472000, 0x3ed0c000}},
      {.blend = enabled | separate | 11 | (12 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0x2f628899,
       .initial = 0x99999999,
       .exported = {0x3ec4a000, 0x3ec4a000, 0x3ec4a000, 0x3e3e6000},
       .constants = {-0.75f, 0.3f, 1.25f, 0.65f}},
      {.blend = enabled | separate | 11 | (12 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0xd453b3dd,
       .initial = 0xdddddddd,
       .exported = {0x3ea5e000, 0x3ea5e000, 0x3ea5e000, 0x3f546000},
       .constants = {-0.75f, 0.3f, 1.25f, 0.65f}},
      {.blend = enabled | separate | 10 | (1 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0x90aeaeae,
       .initial = 0xa5a5a5a5,
       .exported = {0x3dd8a000, 0x3dd8a000, 0x3dd8a000, 0x3f10a000}},
      {.blend = enabled | separate | 10 | (1 << 8) | (1 << 16),
       .mask = 15,
       .expected = 0x73d0d0d0,
       .initial = 0xcccccccc,
       .exported = {0x3db44000, 0x3db44000, 0x3db44000, 0x3ee6a000}},
  };
  for (uint32_t swap = 0; swap < 4; ++swap) {
    const auto memory_word = [swap](uint32_t rgba) {
      const uint32_t r = rgba & 255, g = (rgba >> 8) & 255, b = (rgba >> 16) & 255, a = rgba >> 24;
      switch (swap) {
      case 1:
        return b | (g << 8) | (r << 16) | (a << 24);
      case 2:
        return a | (b << 8) | (g << 16) | (r << 24);
      case 3:
        return a | (r << 8) | (g << 16) | (b << 24);
      default:
        return rgba;
      }
    };
    SCOPED_TRACE(swap);
    for (const auto &test : cases) {
      SCOPED_TRACE(test.blend);
      SCOPED_TRACE(test.expected);
      auto state = rectangle_state();
      auto &context = state.context_registers;
      context[gfx12 ? 0x3b0 : 0x31c] = 10 | (test.number_format << 8) | (swap << 11);
      context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
      context[gfx12 ? 0x31f : 0x3b8] = gfx12 ? 3u << 15 : 26u << 14;
      context[0x318] = 0x1000;
      context[gfx12 ? 0x214 : 0x8e] = test.mask;
      context[gfx12 ? 0x215 : 0x8f] = 15;
      context[gfx12 ? 0x195 : 0x1c5] = test.export_format;
      context[gfx12 ? 0x216 : 0x202] = test.color_control;
      context[0x1e0] = test.blend;
      const uint32_t boundary[] = {0x3b008081, 0x3e56d6d7, 0x3ed5d5d6, 0x3f202020};
      for (uint32_t c = 0; c < 4; ++c)
        context[0x105 + c] =
            test.constant_boundary ? boundary[c] : std::bit_cast<uint32_t>(test.constants[c]);
      context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
          std::bit_cast<uint32_t>(2.0f);
      context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
      memory_.write32(0x100000, memory_word(test.initial));

      auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
      export_rectangle_vertices(*draw);
      if (test.reject) {
        EXPECT_THROW(draw->advance(*access_), std::runtime_error);
        EXPECT_EQ(memory_.read32(0x100000), memory_word(test.initial));
        continue;
      }
      ASSERT_TRUE(draw->advance(*access_));
      initialize_fragment(draw);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
        draw->export_lane(*wave_, lane, 0, test.export_format == 4 ? 3 : 15, test.exported);
      EXPECT_FALSE(draw->advance(*access_));
      EXPECT_EQ(memory_.read32(0x100000), memory_word(test.expected));
    }
  }
}

TEST_P(GraphicsExportTest, LogicOperationsUseConvertedBitsAndLogicalChannelMasks) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t sources[] = {0x3c695ac3, 0x3cc35a69, 0xc35a693c, 0x695ac33c};
  const uint32_t masks[] = {0x00ff00ff, 0x00ff00ff, 0xff00ff00, 0xff00ff00};
  for (uint32_t swap = 0; swap < 4; ++swap) {
    const uint32_t src = sources[swap], dst = 0xf3c596a5;
    const uint32_t results[] = {0,          ~(src | dst), ~src & dst, ~src,
                                src & ~dst, ~dst,         src ^ dst,  ~(src & dst),
                                src & dst,  ~(src ^ dst), dst,        ~src | dst,
                                src,        src | ~dst,   src | dst,  ~0u};
    for (uint32_t rop = 0; rop < 16; ++rop) {
      for (bool disable : {false, true})
        for (uint32_t mask : {5u, 15u}) {
          SCOPED_TRACE(swap);
          SCOPED_TRACE(rop);
          SCOPED_TRACE(mask);
          SCOPED_TRACE(disable);
          auto state = rectangle_state();
          auto &context = state.context_registers;
          context[gfx12 ? 0x3b0 : 0x31c] = 10 | (4u << 8) | (swap << 11);
          context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
          context[gfx12 ? 0x31f : 0x3b8] = gfx12 ? 3u << 15 : 26u << 14;
          context[0x318] = 0x1000;
          context[0x1e0] = disable ? 1u << 31 : 0;
          context[gfx12 ? 0x214 : 0x8e] = mask;
          context[gfx12 ? 0x215 : 0x8f] = 15;
          context[gfx12 ? 0x195 : 0x1c5] = 9u;
          context[gfx12 ? 0x216 : 0x202] = (rop * 0x110000) | 0x10;
          context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
              std::bit_cast<uint32_t>(2.0f);
          context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
          memory_.write32(0x100000, 0xf3c596a5);

          auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
          export_rectangle_vertices(*draw);
          ASSERT_TRUE(draw->advance(*access_));
          initialize_fragment(draw);
          for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
            draw->export_lane(*wave_, lane, 0, 15, {0xc3, 0x5a, 0x69, 0x3c});
          EXPECT_FALSE(draw->advance(*access_));
          const uint32_t bits = mask == 15 ? ~0u : masks[swap];
          EXPECT_EQ(memory_.read32(0x100000),
                    ((disable ? src : results[rop]) & bits) | (dst & ~bits));
        }
    }
  }
}

TEST_P(GraphicsExportTest, ColorAttachmentsWriteFullTexelsAndPreserveMaskedChannels) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    // RGBA32F is the common blend witness; packed formats override this layout.
    uint32_t data_format = 14, number_format = 7, export_format = 9, bytes = 16;
    uint32_t components = 4, mask = 15;
    std::array<uint32_t, 4> exported, expected;
    uint32_t blend = 0, blend_opt = 0;
    std::array<uint32_t, 4> initial{0xcccccccc, 0xcccccccc, 0xcccccccc, 0xcccccccc};
    std::array<uint32_t, 4> constant_bits{0xbf400000, 0x3e99999a, 0x3fa00000, 0x3f266666};
  };
  const Case cases[] = {
      {.data_format = 2,
       .export_format = 4,
       .bytes = 2,
       .components = 1,
       .mask = 1,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0x3400}},
      {.data_format = 5,
       .export_format = 4,
       .bytes = 4,
       .components = 2,
       .mask = 1,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0x38003400}},
      {.data_format = 4,
       .export_format = 3,
       .bytes = 4,
       .components = 1,
       .mask = 3,
       .exported = {0x3f000000, 0x3e800000},
       .expected = {0x3f600000},
       .blend = 0x61000504u,
       .initial = {0x3f800000}},
      {.data_format = 9,
       .number_format = 0,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0xeff80100}},
      {.data_format = 1,
       .number_format = 4,
       .export_format = 7,
       .bytes = 1,
       .components = 1,
       .mask = 3,
       .exported = {0x432100ab, 0x87651234},
       .expected = {0xab}},
      // A zero export bypasses arithmetic and preserves signaling NaN payloads.
      {.exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .expected = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0xff812345u},
       .blend = 0x61000101u,
       .blend_opt = 0x1100111u,
       .initial = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0xff812345u}},
      // Subnormal exports do not trigger zero flags, then flush before arithmetic.
      {.exported = {0x00000001u, 0x00000001u, 0x00000001u, 0x00000000u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x1100111u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u}},
      // Active NaN products canonicalize their result.
      {.exported = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x00000000u},
       .expected = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u},
       .blend = 0x61000101u,
       .blend_opt = 0x1100111u,
       .initial = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0x7fc12345u}},
      // A positive-zero coefficient suppresses a NaN color.
      {.exported = {0x7fc12345u, 0x7fc12345u, 0x7fc12345u, 0x00000000u},
       .expected = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u},
       .blend = 0x61000604u,
       .blend_opt = 0x1100174u,
       .initial = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      // Negative subnormal coefficients flush with their sign intact.
      {.exported = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x80000001u},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u},
       .blend = 0x61000604u,
       .blend_opt = 0x1100174u,
       .initial = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x80000000u}},
      // Saturation selects the numeric factor when the source alpha is NaN.
      {.exported = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x7fc12345u},
       .expected = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u},
       .blend = 0x6100010au,
       .blend_opt = 0x1100176u,
       .initial = {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      // MIN selects the numeric operand instead of a source NaN.
      {.exported = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0x7f800000u},
       .expected = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0xff800000u},
       .blend = 0x61400141u,
       .initial = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0xff800000u}},
      // MAX normalizes zero signs and canonicalizes two NaNs.
      {.exported = {0x80000000u, 0x80000000u, 0x80000000u, 0x7f812345u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0xffc00000u},
       .blend = 0x61600161u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0xffc12345u}},
      // Constant zero/one equations preserve raw destination bits.
      {.exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61000c0bu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      // A subnormal constant flushes without qualifying for raw destination bypass.
      {.exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0xffc00000u, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61000c0bu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u}},
      // Signaling NaN constants produce canonical NaNs during arithmetic.
      {.exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xff800000u},
       .blend = 0x61000c0bu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x7f812345u, 0x7f812345u, 0x7f812345u, 0x7f812345u}},
      // An exact source replacement preserves raw source bits.
      {.mask = 1,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0x7ff339aeu, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61000b0cu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      // Source replacement also bypasses subtraction of a fixed zero.
      {.mask = 1,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0x7ff339aeu, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61200b2cu,
       .blend_opt = 0x2100277u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      // Constant classification includes RGB channels absent from the export.
      {.mask = 1,
       .exported = {0x7ff339aeu, 0x3f800000u, 0x40000000u, 0xea97a4b8u},
       .expected = {0xffc00000u, 0x40400000u, 0x40800000u, 0xff800000u},
       .blend = 0x61000c0bu,
       .blend_opt = 0x1100177u,
       .initial = {0x7f839d69u, 0x40400000u, 0x40800000u, 0xff800000u},
       .constant_bits = {0x00000000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      // Physical zero signs with SX bypass enabled, disabled, and partial exports.
      {.exported = {0, 0, 0, 0},
       .expected = {0, 0, 0, 0},
       .blend = 0x61000604u,
       .blend_opt = 0x01100174u,
       .initial = {0, 0, 0, 0x80000000u}},
      {.exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x80000000u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .blend = 0x61000604u,
       .blend_opt = 0x00000000u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u}},
      {.exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x80000000u},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u},
       .blend = 0x61200624u,
       .blend_opt = 0x00000000u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u}},
      {.exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x01100111u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u}},
      {.exported = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x00000000u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00000000u}},
      {.exported = {0x00000000u, 0x3f800000u, 0x40000000u, 0x00000000u},
       .expected = {0x00000000u, 0x40800000u, 0x40c00000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x01100111u,
       .initial = {0x80000000u, 0x40400000u, 0x40800000u, 0x00000000u}},
      {.mask = 1,
       .exported = {0x00000000u, 0x3f800000u, 0x40000000u, 0x00000000u},
       .expected = {0x80000000u, 0x40400000u, 0x40800000u, 0x00000000u},
       .blend = 0x61000101u,
       .blend_opt = 0x01100111u,
       .initial = {0x80000000u, 0x40400000u, 0x40800000u, 0x00000000u}},
      {.exported = {0x3f800000u, 0x3f800000u, 0x40000000u, 0x00000000u},
       .expected = {0x00000000u, 0x40400000u, 0x41000000u, 0x00000000u},
       .blend = 0x61000204u,
       .blend_opt = 0x01100124u,
       .initial = {0x80000000u, 0x40400000u, 0x40800000u, 0x00000000u}},
      // Saturation retains factor selection, tie behavior, and comparison precision.
      {.exported = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u},
       .expected = {0, 0, 0, 0x3f800000u},
       .blend = 0x6100010au,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x3f800000u}},
      {.exported = {0xbf653334u, 0x3fa10000u, 0xc1023334u, 0x3ebb3333u},
       .expected = {0xc014f348u, 0x3f35bb33u, 0x3cca8ee1u, 0x3f5d999au},
       .blend = 0x4000010au,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},
      {.exported = {0x00800000u, 0x00800000u, 0x00800000u, 0xff7fffffu},
       .expected = {0xc03fffffu, 0xc03fffffu, 0xc03fffffu, 0x7f7fffffu},
       .blend = 0x6100010au,
       .initial = {0x3f800001u, 0x3f800001u, 0x3f800001u, 0x7f7fffffu}},
      {.exported = {0x4b0b53f3u, 0x4b0b53f3u, 0x4b0b53f3u, 0x3f662b48u},
       .expected = {0x3ddb8000u, 0x3ddb8000u, 0x3ddb8000u, 0x3dcea5c0u},
       .blend = 0x6100010au,
       .initial = {0xcafa89f5u, 0xcafa89f5u, 0xcafa89f5u, 0x3dcea5c0u}},
      {.exported = {0x80000000u, 0x80000000u, 0x80000000u, 0x3f800000u},
       .expected = {0x00000000u, 0x00000000u, 0x00000000u, 0x00800000u},
       .blend = 0x6100010au,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0x00800000u}},
      // Physical constant-alpha and mixed inverse-factor cancellation.
      {.exported = {0xaf0060a4u, 0xaf0060a4u, 0xaf0060a4u, 0x4052fde5u},
       .expected = {0x1d800000u, 0x1d800000u, 0x1d800000u, 0x3f7fe89au},
       .blend = 0x61000507u,
       .initial = {0xa8a37ba9u, 0xa8a37ba9u, 0xa8a37ba9u, 0x3f7fe89au}},
      {.exported = {0x80000000u, 0x80000000u, 0x80000000u, 0},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0},
       .blend = 0x61000504u,
       .blend_opt = 0x01100154u,
       .initial = {0x80000000u, 0x80000000u, 0x80000000u, 0}},
      {.exported = {0x3f883334u, 0x3fb1cccdu, 0xbf066667u, 0x4058e666u},
       .expected = {0xbc08f5cfu, 0x3f7d8a3du, 0x3f3570a5u, 0x40182f5cu},
       .blend = 0x40001211u,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},
      {.exported = {0x3ceec6c7u, 0x3ceec6c7u, 0x3ceec6c7u, 0xbe9368dbu},
       .expected = {0xaf2a0000u, 0xaf2a0000u, 0xaf2a0000u, 0x4056d022u},
       .blend = 0x61000705u,
       .initial = {0x3c828093u, 0x3c828093u, 0x3c828093u, 0x4056d022u}},
      {.exported = {0x395addbeu, 0x395addbeu, 0x395addbeu, 0xb07442afu},
       .expected = {0x2cb1e000u, 0x2cb1e000u, 0x2cb1e000u, 0xc0e932bbu},
       .blend = 0x61000705u,
       .initial = {0xb7d34666u, 0xb7d34666u, 0xb7d34666u, 0xc0e932bbu}},
      // Physical RDNA3/4 cancellation, factors, sticky rounding, and underflow.
      {.exported = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0x80000000u},
       .expected = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0x80000000u},
       .blend = 0x61000104u,
       .blend_opt = 0x01100114u,
       .initial = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0x80000000u}},
      {.exported = {0x80800000u, 0x80800000u, 0x80800000u, 0x3f7fffffu},
       .expected = {0x80000000u, 0x80000000u, 0x80000000u, 0x3f800000u},
       .blend = 0x61000604u,
       .initial = {0, 0, 0, 0x3f800000u}},
      {.exported = {0x00800001u, 0x00800001u, 0x00800001u, 0x3f7ffffeu},
       .expected = {0x00800000u, 0x00800000u, 0x00800000u, 0x3f800000u},
       .blend = 0x61000604u,
       .initial = {0, 0, 0, 0x3f800000u}},
      {.exported = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0x40000000u},
       .expected = {0, 0, 0, 0xc0000000u},
       .blend = 0x61000000u,
       .initial = {0xc0000000u, 0xc0000000u, 0xc0000000u, 0xc0000000u}},
      {.exported = {0x3ffbe98eu, 0x3ffbe98eu, 0x3ffbe98eu, 0x3fe52bfcu},
       .expected = {0x336b0000u, 0x336b0000u, 0x336b0000u, 0xc0618334u},
       .blend = 0x61000104u,
       .initial = {0xc0618334u, 0xc0618334u, 0xc0618334u, 0xc0618334u}},
      {.exported = {0x3ffbe98eu, 0x3ffbe98eu, 0x3ffbe98eu, 0x3fe52bfcu},
       .expected = {0x336b0000u, 0x336b0000u, 0x336b0000u, 0x3f800000u},
       .blend = 0x61000604u,
       .initial = {0xc0618334u, 0xc0618334u, 0xc0618334u, 0x3f800000u}},
      {.exported = {0x3fa44755u, 0x3fa44755u, 0x3fa44755u, 0x3fd24517u},
       .expected = {0x32c80000u, 0x32c80000u, 0x32c80000u, 0xc000141eu},
       .blend = 0x61000604u,
       .initial = {0x3f86d9a8u, 0x3f86d9a8u, 0x3f86d9a8u, 0xc000141eu}},
      {.exported = {0xbb33749eu, 0xbb33749eu, 0xbb33749eu, 0x94b0b5edu},
       .expected = {0xf27fbcf7u, 0xf27fbcf7u, 0xf27fbcf7u, 0xd9b9d285u},
       .blend = 0x61000604u,
       .initial = {0x583028fcu, 0x583028fcu, 0x583028fcu, 0xd9b9d285u}},

      // Physical FP16 products align to 23 bits before truncating the sum.
      {.data_format = 12,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0x00c300c3u, 0x002800c3u},
       .expected = {0x44fc44fcu, 0x3a6644fcu},
       .blend = 0x40000501,
       .initial = {0x44fc44fcu, 0x3a6644fcu}},
      // A destination bypass also retains FP16 negative zero.
      {.data_format = 12,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0x3fa23fa2u, 0x00003fa2u},
       .expected = {0x6f106f10u, 0x80006f10u},
       .blend = 0x40000504,
       .blend_opt = 0x01100154u,
       .initial = {0x6f106f10u, 0x80006f10u}},
      // Physical FP16/FP32 blend captures include product precision and RTZ output.
      {.data_format = 12,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0xbff0c3d8u, 0x2c00c7e8u},
       .expected = {0x2f10c03du, 0x379040a3u},
       .blend = 0x40000504,
       .initial = {0x3400c000u, 0x38004200u}},
      {.data_format = 12,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0xbff0c3f8u, 0x0000c7f8u},
       .expected = {0xb6b8b818u, 0x319ac95bu},
       .blend = 0x40000c0b,
       .initial = {0x3400c000u, 0x38004200u}},
      {.exported = {0xbf943334u, 0x3fb1cccdu, 0xc116999au, 0x3d666666u},
       .expected = {0xbff9efaeu, 0x3ea0ce14u, 0x401350a4u, 0x3ef33852u},
       .blend = 0x40000504,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},
      {.exported = {0xbf956667u, 0x3fb1cccdu, 0xc1173334u, 0x3d2cccccu},
       .expected = {0xc027f999u, 0x3f177ae2u, 0xc1490001u, 0x3e4f47afu},
       .blend = 0x40000c0b,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},
      {.exported = {0xbf653334u, 0x3fb1cccdu, 0xc105cccdu, 0x3ee66666u},
       .expected = {0x3f8d6666u, 0x3f91cccdu, 0xc135cccdu, 0xbd4cccd0u},
       .blend = 0x40000121,
       .initial = {0xc0000000u, 0x3e800000u, 0x40400000u, 0x3f000000u}},

      // Full-width float and integer values for R32/RG32 attachments.
      {.data_format = 4,
       .export_format = 1,
       .bytes = 4,
       .components = 1,
       .mask = 1,
       .exported = {0xbf812345, 0x3eabcdef},
       .expected = {0xbf812345, 0x3eabcdef}},
      {.data_format = 4,
       .number_format = 4,
       .export_format = 1,
       .bytes = 4,
       .components = 1,
       .mask = 1,
       .exported = {0x87654321, 0xfedcba98},
       .expected = {0x87654321, 0xfedcba98}},
      {.data_format = 4,
       .number_format = 5,
       .export_format = 1,
       .bytes = 4,
       .components = 1,
       .mask = 1,
       .exported = {0x80000001, 0x7ffffffe},
       .expected = {0x80000001, 0x7ffffffe}},
      {.data_format = 11,
       .export_format = 2,
       .bytes = 8,
       .components = 2,
       .mask = 3,
       .exported = {0xbf812345, 0x3eabcdef},
       .expected = {0xbf812345, 0x3eabcdef}},
      {.data_format = 11,
       .number_format = 4,
       .export_format = 2,
       .bytes = 8,
       .components = 2,
       .mask = 3,
       .exported = {0x87654321, 0xfedcba98},
       .expected = {0x87654321, 0xfedcba98}},
      {.data_format = 11,
       .number_format = 5,
       .export_format = 2,
       .bytes = 8,
       .components = 2,
       .mask = 3,
       .exported = {0x80000001, 0x7ffffffe},
       .expected = {0x80000001, 0x7ffffffe}},

      // FP16 sRGB exports captured on both physical RDNA3 and RDNA4.
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x00000000, 0x00000000},
       .expected = {0x00000000}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x41941234, 0x1234c70c},
       .expected = {0x0000ff03}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x2b132aab, 0x2aabd685},
       .expected = {0x0d004341}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x76c737ff, 0x37ff4471},
       .expected = {0x7fffffbb}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x78003800, 0x38004800},
       .expected = {0x80ffffbc}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x571c3bfc, 0x3bfc75c4},
       .expected = {0xffffffff}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x5c003c00, 0x3c008400},
       .expected = {0xff00ffff}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x9c007c00, 0x7c004400},
       .expected = {0xffff00ff}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0x0e007e00, 0x7e006200},
       .expected = {0x00ff0100}},
      {.data_format = 10,
       .number_format = 6,
       .export_format = 4,
       .bytes = 4,
       .mask = 3,
       .exported = {0xdc00bc00, 0xbc000400},
       .expected = {0x00000000}},
      // Signed packed exports sign-extend integers and normalize signed endpoints.
      {.data_format = 12,
       .number_format = 1,
       .export_format = 6,
       .bytes = 8,
       .mask = 3,
       .exported = {0x80018000, 0x7fff0001},
       .expected = {0x80018001, 0x7fff0001}},
      {.data_format = 12,
       .number_format = 1,
       .export_format = 6,
       .bytes = 8,
       .mask = 3,
       .exported = {0xc001ffff, 0x40010000},
       .expected = {0xc001ffff, 0x40010000}},
      {.data_format = 12,
       .number_format = 5,
       .export_format = 8,
       .bytes = 8,
       .mask = 3,
       .exported = {0x80007fff, 0xffff0001},
       .expected = {0x80007fff, 0xffff0001}},
      {.number_format = 5,
       .export_format = 8,
       .mask = 3,
       .exported = {0x80007fff, 0xffff0001},
       .expected = {0x00007fff, 0xffff8000, 0x00000001, 0xffffffff}},
      // Packed UNORM16 exports retain every channel bit, including endpoints.
      {.data_format = 12,
       .number_format = 0,
       .export_format = 5,
       .bytes = 8,
       .mask = 3,
       .exported = {0x00010000, 0xfffffffe},
       .expected = {0x00010000, 0xfffffffe}},
      {.data_format = 12,
       .number_format = 0,
       .export_format = 5,
       .bytes = 8,
       .mask = 3,
       .exported = {0x7fff8000, 0xabc12345},
       .expected = {0x7fff8000, 0xabc12345}},
      {.data_format = 12,
       .number_format = 0,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0x80004000, 0xffffbfff}},
      {.data_format = 12,
       .export_format = 4,
       .bytes = 8,
       .mask = 3,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0x38003400, 0x3c003a00}},
      {.data_format = 12,
       .number_format = 4,
       .export_format = 7,
       .bytes = 8,
       .mask = 3,
       .exported = {0x45670123, 0xcdef89ab},
       .expected = {0x45670123, 0xcdef89ab}},
      // RGB-only and sparse component exports preserve the remaining channels.
      {.mask = 7,
       .exported = {0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000},
       .expected = {0x3e800000, 0x3f000000, 0x3f400000, 0xcccccccc}},
      {.mask = 10,
       .exported = {0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000},
       .expected = {0xcccccccc, 0x3f000000, 0xcccccccc, 0x3f800000}},
      {.data_format = 12,
       .export_format = 4,
       .bytes = 8,
       .mask = 1,
       .exported = {0x38003400, 0x3c003a00},
       .expected = {0x38003400, 0xcccccccc}},
      {.exported = {0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000},
       .expected = {0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000}},
      {.number_format = 4,
       .exported = {0x12345678, 0x87654321, 0x0fedcba9, 0x9abcdef0},
       .expected = {0x12345678, 0x87654321, 0x0fedcba9, 0x9abcdef0}},
  };
  for (const auto &test : cases) {
    for (uint32_t write_mask : {1u, 2u, 5u, 15u}) {
      SCOPED_TRACE(test.data_format);
      SCOPED_TRACE(test.number_format);
      SCOPED_TRACE(write_mask);
      auto state = rectangle_state();
      auto &context = state.context_registers;
      context[gfx12 ? 0x3b0 : 0x31c] = test.data_format | (test.number_format << 8);
      context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
      // Linear targets cover the RGBA32F linear-image clear CTS regression.
      // META_LINEAR is immaterial on GFX11 when no metadata is enabled.
      context[gfx12 ? 0x31f : 0x3b8] = !gfx12 && write_mask == 15 ? 1u << 13 : 0;
      context[gfx12 ? 0x31b : 0x31d] = test.components < 4 ? 1u << 2 : 0; // FORCE_DST_ALPHA_1
      context[0x318] = 0x1000;
      context[0x1e0] = test.blend;
      context[0x1d8] = test.blend_opt;
      for (uint32_t c = 0; c < 4; ++c)
        context[0x105 + c] = test.constant_bits[c];
      context[gfx12 ? 0x214 : 0x8e] = write_mask;
      context[gfx12 ? 0x215 : 0x8f] = 15;
      context[gfx12 ? 0x195 : 0x1c5] = test.export_format;
      context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
      context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
          std::bit_cast<uint32_t>(2.0f);
      context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
      for (uint32_t offset = 0; offset <= ((test.bytes + 3) & ~3u); offset += 4)
        memory_.write32(0x100000 + offset,
                        offset < test.bytes ? test.initial[offset / 4] : 0xcccccccc);

      auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
      export_rectangle_vertices(*draw);
      ASSERT_TRUE(draw->advance(*access_));
      initialize_fragment(draw);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
        draw->export_lane(*wave_, lane, 0, test.mask, test.exported);
      EXPECT_FALSE(draw->advance(*access_));
      for (uint32_t byte = 0; byte < test.bytes; ++byte) {
        const uint32_t component = byte / (test.bytes / test.components);
        uint8_t expected =
            ((write_mask & (1u << component)) ? test.expected[byte / 4] : test.initial[byte / 4]) >>
            (8 * (byte % 4));
        if (test.data_format == 9) {
          const uint32_t bits =
              ((write_mask & 1) ? 0x3ffu : 0) | ((write_mask & 2) ? 0xffc00u : 0) |
              ((write_mask & 4) ? 0x3ff00000u : 0) | ((write_mask & 8) ? 0xc0000000u : 0);
          expected = ((test.expected[0] & bits) | (test.initial[0] & ~bits)) >> (8 * byte);
        }
        uint8_t actual = 0;
        ASSERT_EQ(access_->read(0x100000 + byte, {reinterpret_cast<std::byte *>(&actual), 1}),
                  amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(actual, expected) << byte;
      }
      EXPECT_EQ(memory_.read32(0x100000 + ((test.bytes + 3) & ~3u)), 0xcccccccc);
    }
  }
}

TEST_P(GraphicsExportTest, MultipleAttachmentsKeepFormatsMasksAndBlendStateIndependent) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool disabled_first : {false, true})
    for (bool sparse : {false, true}) {
      for (bool blend : {false, true}) {
        SCOPED_TRACE(testing::Message() << "sparse=" << sparse << " disabled_first="
                                        << disabled_first << " blend=" << blend);
        const std::array<uint32_t, 3> targets =
            sparse ? std::array<uint32_t, 3>{1, 3, 7} : std::array<uint32_t, 3>{0, 1, 2};
        const std::array<uint32_t, 3> bytes{4, 8, 16};
        const std::array<uint32_t, 3> formats{10u, 12u | (7u << 8), 14u | (7u << 8)};
        const std::array<uint32_t, 3> export_formats{4u, 4u, 9u};
        const std::array<uint32_t, 3> masks{disabled_first ? 0u : 15u, 5, 10};
        const std::array<std::array<uint32_t, 4>, 3> expected{
            std::array<uint32_t, 4>{blend || disabled_first ? 0xccccccccu : 0xff0000ffu},
            std::array<uint32_t, 4>{0xcccc3400, 0xcccc3a00},
            std::array<uint32_t, 4>{0xcccccccc, 0x40400000, 0xcccccccc, 0x40a00000}};
        auto state = rectangle_state();
        auto &context = state.context_registers;
        for (uint32_t i = 0; i < targets.size(); ++i) {
          const uint32_t target = targets[i], block = 0x318 + (gfx12 ? 9 : 15) * target;
          context[block] = 0x1000 + 0x100 * i;
          context[gfx12 ? 0x3b0 + target : block + 4] = formats[i];
          context[gfx12 ? block + 6 : 0x3b0 + target] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
          context[gfx12 ? 0x214 : 0x8e] |= masks[i] << (4 * target);
          context[gfx12 ? 0x195 : 0x1c5] |= export_formats[i] << (4 * i);
          context[gfx12 ? 0x215 : 0x8f] |= 15u << (4 * target);
          // ZERO * source + ONE * destination on only the first target.
          context[0x1e0 + target] = blend && i == 0 ? (1u << 30) | (1 << 8) : 0;
          for (uint32_t offset = 0; offset <= bytes[i]; offset += 4)
            memory_.write32(0x100000 + 0x10000 * i + offset, 0xcccccccc);
        }
        context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
        context[0x10f] = context[0x110] = context[0x111] = context[0x112] =
            std::bit_cast<uint32_t>(2.0f);
        context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
        // LESS plus a depth write must run once, before all color attachments.
        context[gfx12 ? 0x1c : 0x200] = 6 | (1 << 4);
        context[gfx12 ? 5 : 7] = 3 | (3 << 16);
        context[gfx12 ? 6 : 0x10] = 3 | ((gfx12 ? 3 : 24) << 4);
        context[gfx12 ? 8 : 0x12] = context[gfx12 ? 10 : 0x14] = 0x2000;
        context[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(1.0f);
        context[0x113] = std::bit_cast<uint32_t>(1.0f);
        memory_.write32(0x200000, std::bit_cast<uint32_t>(1.0f));
        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        for (uint32_t i = 0; i < 3; ++i)
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(1.0f)});
        draw->export_lane(*wave_, 0, 20, 1,
                          {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
        const auto dispatch = draw->advance(*access_);
        ASSERT_TRUE(dispatch);
        EXPECT_EQ(amdgpu::GraphicsDrawTestAccess::export_storage(*draw)[0],
                  size_t(dispatch->total_wgs) * 64 * (disabled_first ? 2 : 3));
        initialize_fragment(draw);
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
          draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000});
          draw->export_lane(*wave_, lane, 1, 3, {0x38003400, 0x3c003a00});
          draw->export_lane(*wave_, lane, 2, 15, {0x40000000, 0x40400000, 0x40800000, 0x40a00000});
        }
        EXPECT_FALSE(draw->advance(*access_));
        EXPECT_EQ(memory_.read32(0x200000), std::bit_cast<uint32_t>(0.5f));
        for (uint32_t i = 0; i < targets.size(); ++i) {
          for (uint32_t word = 0; word < bytes[i] / 4; ++word)
            EXPECT_EQ(memory_.read32(0x100000 + 0x10000 * i + 4 * word), expected[i][word])
                << "target " << targets[i] << " word " << word;
          EXPECT_EQ(memory_.read32(0x100000 + 0x10000 * i + bytes[i]), 0xcccccccc);
        }
      }
    }
}

TEST_P(GraphicsExportTest, MultipleAttachmentViewsClipLayersIndependently) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  auto state = rectangle_state();
  auto &context = state.context_registers;
  const std::array<uint32_t, 2> targets{0, 7};
  const std::array<uint32_t, 2> views{0, 1 | (3u << (gfx12 ? 14 : 13))};
  for (uint32_t i = 0; i < targets.size(); ++i) {
    const uint32_t target = targets[i], block = 0x318 + (gfx12 ? 9 : 15) * target;
    context[block] = (0x100000 + i * 0x100000) >> 8;
    context[gfx12 ? 0x3b0 + target : block + 4] = 10;
    context[gfx12 ? block + 6 : 0x3b0 + target] = gfx12 ? 3 | (3 << 16) : 3 | (3 << 14);
    context[gfx12 ? block + 7 : 0x3b8 + target] = (gfx12 ? 3u << 15 : 26u << 14) | 3;
    // Target 0 has one layer; target 7 starts at layer 1 and has three layers.
    context[gfx12 ? block + 1 : block + 3] = views[i];
    context[gfx12 ? 0x214 : 0x8e] |= 15u << (4 * target);
    context[gfx12 ? 0x215 : 0x8f] |= 15u << (4 * target);
    context[gfx12 ? 0x195 : 0x1c5] |= 4u << (4 * i);
    for (uint32_t layer = 0; layer < 4; ++layer)
      memory_.write32(
          amdgpu::image_layer_base(gfx12, 0x100000 + i * 0x100000, 65536, layer, 4, gfx12 ? 3 : 26),
          0xcccccccc);
  }
  context[gfx12 ? 0x206 : 0x207] = 1u << 18;
  context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
  context[0x10f] = context[0x110] = context[0x111] = context[0x112] = std::bit_cast<uint32_t>(2.0f);
  context[0x91] = gfx12 ? 0 : 1 | (1 << 16);
  auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
  for (uint32_t i = 0; i < 3; ++i) {
    draw->export_lane(*wave_, i, 12, 15,
                      {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                       std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f), 0,
                       std::bit_cast<uint32_t>(1.0f)});
    draw->export_lane(*wave_, i, 13, 4, {0, 0, 1, 0});
  }
  draw->export_lane(*wave_, 0, 20, 1,
                    {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
  ASSERT_TRUE(draw->advance(*access_));
  initialize_fragment(draw);
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
    draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000});
    draw->export_lane(*wave_, lane, 1, 3, {0x3c000000, 0x3c000000});
  }
  EXPECT_FALSE(draw->advance(*access_));
  for (uint32_t i = 0; i < targets.size(); ++i)
    for (uint32_t layer = 0; layer < 4; ++layer) {
      const uint64_t address =
          amdgpu::image_layer_base(gfx12, 0x100000 + i * 0x100000, 65536, layer, 4, gfx12 ? 3 : 26);
      EXPECT_EQ(memory_.read32(address), targets[i] == 7 && layer == 2 ? 0xff00ff00 : 0xcccccccc)
          << "target " << targets[i] << " layer " << layer;
    }
}

TEST_P(GraphicsExportTest, ColorAttachmentWritesSelectedMipAndPreservesOtherLevels) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t base = 0x400000;
  const auto address = [&](uint32_t level, uint32_t slice) {
    const auto mip = amdgpu::image_mip_layout(gfx12, gfx12 ? 3 : 27, 4, 200, 180, 8, level);
    return (base + kMipLastTexelOffsets[gfx12][level] + slice * mip->slice_size) ^
           (gfx12 ? 0 : amdgpu::gfx11_image_slice_xor(slice, 4, 27));
  };
  for (bool volume : {false, true})
    for (bool compressed : {false, true}) {
      if (gfx12 && compressed)
        continue;
      for (uint32_t level = 0; level < 8; ++level) {
        SCOPED_TRACE(testing::Message() << "level=" << level << " volume=" << volume);
        const uint32_t slice = volume && level < 7 ? 1 : 0;
        const uint32_t width = std::max(1u, 200u >> level), height = std::max(1u, 180u >> level);
        for (uint32_t mip = 0; mip < 8; ++mip)
          for (uint32_t z = 0; z <= (volume && mip < 7 ? 1u : 0u); ++z)
            memory_.write32(address(mip, z), 0x12345678);
        auto state = rectangle_state();
        auto &context = state.context_registers;
        context[gfx12 ? 0x3b0 : 0x31c] = 10;
        context[gfx12 ? 0x31e : 0x3b0] = gfx12 ? 179 | (199 << 16) : 179 | (199 << 14) | (7u << 28);
        context[gfx12 ? 0x31f : 0x3b8] = gfx12 ? (3u << 15) | (7u << 19) : 27u << 14;
        context[gfx12 ? 0x31a : 0x31b] = gfx12 ? level : level << 26;
        if (volume) {
          context[gfx12 ? 0x31f : 0x3b8] |= (2u << 24) | 127;
          context[gfx12 ? 0x319 : 0x31b] |= slice | (slice << (gfx12 ? 14 : 13));
        }
        context[0x318] = base >> 8;
        context[gfx12 ? 0x214 : 0x8e] = 15;
        context[gfx12 ? 0x215 : 0x8f] = 15;
        context[gfx12 ? 0x195 : 0x1c5] = 4;
        context[gfx12 ? 0x216 : 0x202] = 0xcc0010;
        context[0x10f] = context[0x110] = std::bit_cast<uint32_t>(width / 2.0f);
        context[0x111] = context[0x112] = std::bit_cast<uint32_t>(height / 2.0f);
        context[0x113] = context[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(1.0f);
        context[0x90] = (width - 1) | ((height - 1) << 16);
        context[0x91] = (width - gfx12) | ((height - gfx12) << 16);

        constexpr uint64_t metadata = 0x200000;
        if (compressed) {
          context[0x31e] |= 1u << 22;
          context[0x3b8] |= 1u << 30;
          context[0x325] = metadata >> 8;
          for (uint32_t i = 0; i < (volume ? 98304u : 49152u); ++i)
            memory_.write8(metadata + i, 0);
        }
        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        export_rectangle_vertices(*draw);
        ASSERT_TRUE(draw->advance(*access_));
        initialize_fragment(draw);
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
          draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000, 0, 0});
        EXPECT_FALSE(draw->advance(*access_));
        if (compressed) {
          for (uint32_t mip = 0; mip < 3; ++mip)
            for (uint32_t z = 0; z <= uint32_t(volume); ++z) {
              const uint64_t key = (metadata + (2 - mip) * 16384 + z * 49152) ^
                                   (amdgpu::gfx11_image_slice_xor(z, 4, 27) & 16383);
              EXPECT_EQ(memory_.read8(key), mip == level && z == slice ? 255 : 0);
            }
        }
        for (uint32_t mip = 0; mip < 8; ++mip)
          for (uint32_t z = 0; z <= (volume && mip < 7 ? 1u : 0u); ++z)
            EXPECT_EQ(memory_.read32(address(mip, z)),
                      mip == level && z == slice ? 0xff0000ffu : 0x12345678u);
      }
    }
}

TEST_P(GraphicsExportTest, HardwareSampleClampsCoordinatesAndConvertsSrgb) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const std::array<uint32_t, 8> descriptor{
      0x1000, (66u << (gfx12 ? 17 : 20)) | (1u << 30), 1u << 14, (9u << 28) | 0xfac, 0, 0, 0, 0};
  for (uint32_t r = 0; r < descriptor.size(); ++r)
    wave_->debug_write_sgpr(8 + r, descriptor[r]);
  for (uint32_t comparison = 0; comparison < 8; ++comparison) {
    SCOPED_TRACE(comparison);
    for (uint32_t r = 4; r < 8; ++r)
      wave_->debug_write_sgpr(r, r == 4 ? 2 | (2 << 3) | (2 << 6) | (comparison << 12) : 0);
    wave_->set_exec(5);
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      wave_->debug_write_vgpr(8, lane, std::bit_cast<uint32_t>(lane == 2 ? 0.1f : 2.0f));
      wave_->debug_write_vgpr(9, lane, std::bit_cast<uint32_t>(lane == 2 ? -0.25f : 2.0f));
      wave_->debug_write_vgpr(10, lane, 0xdeadbeef);
    }
    memory_.write32(0x100000, 0xff000000);
    memory_.write32(gfx12 ? 0x100084 : 0x100104, 0x804080ff);
    // The cube shader aliases both coordinate VGPRs with the sampled result.
    std::array<uint32_t, 4> words{0xe7c6c001, 0x02001008, 0x00000809, 0};
    if (!gfx12) {
      const auto mimg = rdna3::build_mimg(
          31, {.nsa = 1, .dim = 1, .dmask = 15, .vaddr = 8, .vdata = 8, .srsrc = 2, .ssamp = 1});
      std::copy(mimg.begin(), mimg.end(), words.begin());
      words[2] = 9;
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(instruction->is_memory_op());
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    ASSERT_NE(instruction->data(), nullptr);
    EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->wait_counter_type,
              gfx12 ? amdgpu::WaitCounterType::SAMPLECNT : amdgpu::WaitCounterType::LOADCNT);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(wave_->debug_read_vgpr(8, 0)), 1.0f);
    EXPECT_EQ(wave_->debug_read_vgpr(9, 0), 0x3e5d0000u);
    EXPECT_EQ(wave_->debug_read_vgpr(10, 0), 0x3d520000u);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(wave_->debug_read_vgpr(11, 0)), 128.0f / 255.0f);
    EXPECT_EQ(wave_->debug_read_vgpr(10, 1), 0xdeadbeef);
    EXPECT_EQ(wave_->debug_read_vgpr(8, 2), 0);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(wave_->debug_read_vgpr(11, 2)), 1.0f);
  }
}

TEST_P(GraphicsExportTest, HardwareSampleFiltersAndAddressesUnorm8) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const std::array<uint32_t, 8> descriptor{
      0x1000, (42u << (gfx12 ? 17 : 20)) | (1u << 30), 1u << 14, (9u << 28) | 0xfac, 0, 0, 0, 0};
  for (uint32_t r = 0; r < descriptor.size(); ++r)
    wave_->debug_write_sgpr(8 + r, descriptor[r]);
  memory_.write32(0x100000, 0x40fa0b00);
  memory_.write32(0x100004, 0x80c94dff);
  memory_.write32(gfx12 ? 0x100080 : 0x100100, 0xc0618eaa);
  memory_.write32(gfx12 ? 0x100084 : 0x100104, 0xff02db55);
  constexpr std::array<uint32_t, 4> first{0, 0x3d30b0b1, 0x3f7afafb, 0x3e808081};
  constexpr std::array<uint32_t, 4> second{0x3f800000, 0x3e9a9a9b, 0x3f49c9ca, 0x3f008081};
  struct Case {
    const char *name;
    uint32_t wrap;
    bool linear, unnormalized;
    uint32_t border;
    float u, v;
    std::array<uint32_t, 4> expected;
    bool srgb = false;
    bool truncate_coordinates = false;
  };
  // Filtering results were captured with the same 2x2 texture on physical
  // gfx1100 and gfx1201. Compare raw floats, including the fractional precision.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const Case cases[] = {
      {"NaN address", 2, true, false, 0, nan, 0.25f, first},
      {"positive infinite clamp", 2, true, false, 0, infinity, 0.25f, second},
      {"negative infinite clamp", 2, true, false, 0, -infinity, 0.25f, first},
      {"infinite repeat", 0, false, false, 0, infinity, 0.25f, first},
      {"infinite border", 6, false, false, 0, infinity, 0.25f, {0, 0, 0, 0}},
      {"center", 2, true, false, 0, 0.5f, 0.5f, {0x3f000000, 0x3ee16161, 0x3f0a0a0a, 0x3f206060}},
      {"fractional",
       2,
       true,
       false,
       0,
       0.25f + 1.0f / 512,
       0.25f,
       {0x3b800000, 0x3d34d4d5, 0x3f7ac9ca, 0x3e810101}},
      {"two fractional axes",
       2,
       true,
       false,
       0,
       0.25f + 255.0f / 512,
       0.25f + 255.0f / 512,
       {0x3eaca808, 0x3f5b0008, 0x3c4a3e3e, 0x3f7f4141}},
      {"fraction rounds even down", 2, true, false, 0, 0.25f + 1.0f / 1024, 0.25f, first},
      {"unnormalized center",
       2,
       true,
       true,
       0,
       1,
       1,
       {0x3f000000, 0x3ee16161, 0x3f0a0a0a, 0x3f206060}},
      {"nearest repeat", 0, false, false, 0, 1.25f, 0.25f, first},
      {"nearest rounded texel boundary", 2, false, false, 0, 0.5f - 1.0f / 1024, 0.25f, second},
      {"nearest truncated texel boundary", 2, false, false, 0, 0.5f - 1.0f / 1024, 0.25f, first,
       false, true},
      {"nearest below rounding boundary", 2, false, false, 0, 0.5f - 3.0f / 2048, 0.25f, first},
      {"nearest rounded border boundary", 6, false, false, 0, -1.0f / 2048, 0.25f, first},
      {"nearest truncated border boundary",
       6,
       false,
       false,
       0,
       -1.0f / 2048,
       0.25f,
       {0, 0, 0, 0},
       false,
       true},
      {"linear repeat", 0, true, false, 0, 1.25f, 0.25f, first},
      {"negative repeat", 0, true, false, 0, -0.25f, 0.25f, second},
      {"mirror repeat", 1, true, false, 0, 1.25f, 0.25f, second},
      {"negative mirror repeat", 1, false, false, 0, -0.25f, 0.25f, first},
      {"mirror once", 3, true, false, 0, -0.75f, 0.25f, second},
      {"edge clamp", 2, true, false, 0, -0.25f, 0.25f, first},
      {"transparent border", 6, false, false, 0, -0.25f, 0.25f, {0, 0, 0, 0}},
      {"opaque black border", 6, true, false, 1, -0.25f, 0.25f, {0, 0, 0, 0x3f800000}},
      {"white border",
       6,
       true,
       false,
       2,
       1.5f,
       0.25f,
       {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000}},
      {"partial border",
       6,
       true,
       false,
       0,
       0,
       0.25f,
       {0, std::bit_cast<uint32_t>(5.5f / 255), std::bit_cast<uint32_t>(125.0f / 255),
        std::bit_cast<uint32_t>(32.0f / 255)}},
      {"unorm normalization boundary",
       2,
       true,
       false,
       0,
       0.25f + 151.0f / 512,
       0.25f,
       {0x3f170000, 0x3e488080, 0x3f5df6f7, 0x3ecc4c4c}},
      {"unorm two-axis normalization boundary",
       2,
       true,
       false,
       0,
       0.25f + 5.0f / 512,
       0.25f + 1.0f / 512,
       {0x3cb48080, 0x3d4da121, 0x3f796a83, 0x3e8403f4}},
      {"srgb texel center",
       2,
       true,
       false,
       0,
       0.25f,
       0.25f,
       {0, 0x3b5b0000, 0x3f750000, 0x3e808081},
       true},
      {"srgb edge alignment",
       2,
       true,
       false,
       0,
       0.25f + 1.0f / 512,
       0.25f,
       {0x3b800000, 0x3b6c2600, 0x3f74a100, 0x3e810101},
       true},
      {"srgb four texels",
       2,
       true,
       false,
       0,
       0.5f,
       0.5f,
       {0x3ebf2000, 0x3e86e800, 0x3ed4e000, 0x3f206060},
       true},
      {"srgb unequal weights",
       2,
       true,
       false,
       0,
       0.25f + 255.0f / 512,
       0.25f + 255.0f / 512,
       {0x3dc3b982, 0x3f33ee5e, 0x3b54a080, 0x3f7f4141},
       true},
  };
  for (uint32_t channels : {1u, 2u, 4u}) {
    SCOPED_TRACE(channels);
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    const uint32_t format = channels == 1 ? 1 : channels == 2 ? 14 : 42;
    const std::array<uint32_t, 4> texels{0x40fa0b00, 0x80c94dff, 0xc0618eaa, 0xff02db55};
    for (uint32_t i = 0; i < 4; ++i) {
      const uint64_t address = 0x100000 + (i / 2) * (gfx12 ? 128 : 256) + (i % 2) * channels;
      ASSERT_EQ(access_->write(address, std::as_bytes(std::span{&texels[i], 1}).first(channels)),
                amdgpu::VmAccessOutcome::Complete);
    }
    for (const auto &test : cases) {
      if (test.srgb && channels != 4)
        continue;
      SCOPED_TRACE(test.name);
      wave_->debug_write_sgpr(9, ((test.srgb ? 66u : format) << (gfx12 ? 17 : 20)) | (1u << 30));
      wave_->debug_write_sgpr(4, test.wrap | (test.wrap << 3) | (2 << 6) |
                                     (uint32_t(test.unnormalized) << 15) |
                                     (uint32_t(test.truncate_coordinates) << 27));
      wave_->debug_write_sgpr(5, 0);
      wave_->debug_write_sgpr(6, test.linear ? (1 << 20) | (1 << 22) : 0);
      wave_->debug_write_sgpr(7, test.border << 30);
      wave_->set_exec(5);
      for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
        wave_->debug_write_vgpr(8, lane, std::bit_cast<uint32_t>(test.u));
        wave_->debug_write_vgpr(9, lane, std::bit_cast<uint32_t>(test.v));
        wave_->debug_write_vgpr(10, lane, 0xdeadbeef);
        wave_->debug_write_vgpr(11, lane, 0xdeadbeef);
      }
      std::array<uint32_t, 4> words{0xe7c6c001, 0x02001008, 0x00000908, 0};
      if (!gfx12) {
        const auto mimg = rdna3::build_mimg(
            31, {.nsa = 1, .dim = 1, .dmask = 15, .vaddr = 8, .vdata = 8, .srsrc = 2, .ssamp = 1});
        std::copy(mimg.begin(), mimg.end(), words.begin());
        words[2] = 9;
      }
      auto decoded = decoder_->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      ASSERT_FALSE(wave_->instruction_execution_failed());
      ASSERT_NE(instruction->data(), nullptr);
      EXPECT_EQ(instruction->data_as<amdgpu::VectorMemState>()->wait_counter_type,
                gfx12 ? amdgpu::WaitCounterType::SAMPLECNT : amdgpu::WaitCounterType::LOADCNT);
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
      pipeline.issue(instruction.release(), *wave_);
      for (uint32_t c = 0; c < channels; ++c) {
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 0), test.expected[c]) << c;
        EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 2), test.expected[c]) << c;
      }
      EXPECT_EQ(wave_->debug_read_vgpr(8, 1), std::bit_cast<uint32_t>(test.u));
      EXPECT_EQ(wave_->debug_read_vgpr(9, 1), std::bit_cast<uint32_t>(test.v));
      EXPECT_EQ(wave_->debug_read_vgpr(10, 1), 0xdeadbeefu);
      EXPECT_EQ(wave_->debug_read_vgpr(11, 1), 0xdeadbeefu);
    }
  }
}

TEST_P(GraphicsExportTest, PackedTenBitFilteringAndAlphaExpansionMatchHardware) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const std::array<uint32_t, 8> descriptor{
      0x1000, (36u << (gfx12 ? 17 : 20)) | (3u << 30), 1 | (7u << 14), (9u << 28) | 0xfac, 0, 0, 0,
      0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  for (uint32_t y = 0; y < 8; ++y)
    for (uint32_t x = 0; x < 8; ++x) {
      const uint32_t packed = ((x * 177 + y * 39) & 1023) | (((x * 39 + y * 91) & 1023) << 10) |
                              (((x * 50 + y * 150) & 1023) << 20) | (((x + y) & 3) << 30);
      memory_.write32(0x100000 + y * (gfx12 ? 128 : 256) + x * 4, packed);
    }
  wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (1 << 20) | (1 << 22));
  wave_->debug_write_sgpr(7, 0);
  // Raw FP32 readbacks from physical GFX11 and GFX12, including normalization
  // boundaries and fractional two-bit alpha that must expand before filtering.
  struct Sample {
    uint32_t index;
    std::array<uint32_t, 4> expected;
  };
  const Sample samples[] = {
      {16, {0x3a312c4b, 0x391c2708, 0x3948320c, 0x3aaaaaab}},
      {4176, {0x3eb24992, 0x3d9dc772, 0x3dca8aa3, 0x3f2bfbff}},
      {32000, {0x3e059565, 0x3e9bd8f6, 0x3f00721d, 0x3f140000}},
      {56083, {0x3e850241, 0x3f120982, 0x3f114752, 0x3f464bff}},
      {65535, {0x3ef43d0f, 0x3f63b8ee, 0x3ebc2f0c, 0x3f2aaaab}},
  };
  wave_->set_exec(1);
  for (const auto &test : samples) {
    SCOPED_TRACE(test.index);
    wave_->debug_write_vgpr(0, 0, std::bit_cast<uint32_t>(float(test.index % 256) / 255));
    wave_->debug_write_vgpr(1, 0, std::bit_cast<uint32_t>(float(test.index / 256) / 255));
    std::array<uint32_t, 4> words{};
    if (gfx12) {
      const auto encoded = rdna4::build_vsample(
          31, {.dim = 1, .dmask = 15, .vdata = 8, .rsrc = 8, .samp = 4, .vaddr0 = 0, .vaddr1 = 1});
      std::copy(encoded.begin(), encoded.end(), words.begin());
    } else {
      const auto encoded = rdna3::build_mimg(
          31, {.dim = 1, .dmask = 15, .vaddr = 0, .vdata = 8, .srsrc = 2, .ssamp = 1});
      std::copy(encoded.begin(), encoded.end(), words.begin());
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    for (uint32_t channel = 0; channel < 4; ++channel)
      EXPECT_EQ(wave_->debug_read_vgpr(8 + channel, 0), test.expected[channel]) << channel;
  }
}

TEST_P(GraphicsExportTest, HardwareSampleFiltersFp16AndMipLevels) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    const char *name;
    std::array<std::array<uint16_t, 4>, 4> texels;
    std::array<uint16_t, 4> mip;
    std::array<std::array<uint32_t, 4>, 6> expected;
  };
  // Raw outputs from the same 2x2 texture and 1x1 mip on physical gfx1100/gfx1201.
  const Case cases[] = {
      {.name = "positive finite",
       .texels = {{{0x0000, 0x3801, 0x3c01, 0x0001},
                   {0x3c00, 0x2001, 0x5bff, 0x03ff},
                   {0x3bff, 0x1001, 0x7bff, 0x0400},
                   {0x3401, 0x2c01, 0x4c10, 0x0801}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x00000000, 0x3f002000, 0x3f802000, 0x33800000},
                     {0x3e7fe000, 0x3ec04000, 0x467fe000, 0x37806000},
                     {0x3ec7f400, 0x3e92f000, 0x4640a000, 0x38002000},
                     {0x3d002000, 0xbe802000, 0x46802100, 0x39802400},
                     {0x3e67fc00, 0xbeb6c800, 0x46b04800, 0x39882200},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "signed finite",
       .texels = {{{0x0000, 0xb801, 0x3c01, 0x8001},
                   {0xbc00, 0x2001, 0xdbff, 0x03ff},
                   {0xbbff, 0x1001, 0xfbff, 0x0400},
                   {0x3401, 0xac01, 0x4c10, 0x8801}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x00000000, 0xbf002000, 0x3f802000, 0xb3800000},
                     {0xbe7fe000, 0xbec02000, 0xc67fe000, 0x377f4000},
                     {0xbeb7f400, 0xbe915800, 0xc6409800, 0x377f2000},
                     {0x3d002000, 0xbf403000, 0x46802100, 0x39801c00},
                     {0xbe17ec00, 0xbf247600, 0x461ff400, 0x39841c80},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "NaN and infinity",
       .texels = {{{0x7c00, 0xfc00, 0x7c01, 0x7e01},
                   {0x3c00, 0x3c00, 0x3c00, 0x3c00},
                   {0x4000, 0xfc00, 0x7c00, 0x0000},
                   {0x7c00, 0x7c00, 0xfc00, 0xfc01}}},
       .mip = {0xfc00, 0x7c00, 0x7e00, 0x3c00},
       .expected = {{{0x7f800000, 0xff800000, 0xffc00000, 0xffc00000},
                     {0x7f800000, 0xff800000, 0xffc00000, 0xffc00000},
                     {0x7f800000, 0xffc00000, 0xffc00000, 0xffc00000},
                     {0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000},
                     {0xffc00000, 0xffc00000, 0xffc00000, 0xffc00000},
                     {0xff800000, 0x7f800000, 0xffc00000, 0x3f800000}}}},
      {.name = "mixed signed zeros",
       .texels = {{{0x0000, 0x8000, 0x0000, 0x8000},
                   {0x8000, 0x8000, 0x0000, 0x0000},
                   {0x0000, 0x0000, 0x0000, 0x8000},
                   {0x8000, 0x0000, 0x0000, 0x8000}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x00000000, 0x80000000, 0x00000000, 0x80000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x3d002000, 0xbf002000, 0x46802000, 0x39802000},
                     {0x3d002000, 0xbf002000, 0x46802000, 0x39802000},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "subnormals",
       .texels = {{{0x0001, 0x0002, 0x0003, 0x03ff},
                   {0x0003, 0x003f, 0x0081, 0x0101},
                   {0x0011, 0x0012, 0x0103, 0x0201},
                   {0x0201, 0x0245, 0x0000, 0x8001}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x33800000, 0x34000000, 0x34400000, 0x387fc000},
                     {0x34a00000, 0x34c00000, 0x36860000, 0x385fe000},
                     {0x36118000, 0x36528000, 0x3694e000, 0x3833f000},
                     {0x3d002008, 0xbf001fff, 0x46802000, 0x39901c00},
                     {0x3d002123, 0xbf001fe6, 0x46802000, 0x398b5f00},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "large cancellation",
       .texels = {{{0x7bff, 0xfbff, 0x7bff, 0x7001},
                   {0xfbff, 0x7bff, 0x03ff, 0xf001},
                   {0x0001, 0x4000, 0xfbff, 0x4001},
                   {0x4000, 0xc001, 0x3c01, 0xc001}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x477fe000, 0xc77fe000, 0x477fe000, 0x46002000},
                     {0x473fe800, 0xc73fe800, 0x46ffe000, 0x45c03000},
                     {0x46bfe800, 0xc6bfe800, 0x46bfe800, 0x45403000},
                     {0x46ffe010, 0xc6ffe100, 0x47400000, 0x45802000},
                     {0x463fe820, 0xc63fea00, 0x46e01400, 0x44c03002},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
      {.name = "negative zeros",
       .texels = {{{0x8000, 0x8000, 0x8000, 0x8000},
                   {0x8000, 0x8000, 0x8000, 0x8000},
                   {0x8000, 0x8000, 0x8000, 0x8000},
                   {0x8000, 0x8000, 0x8000, 0x8000}}},
       .mip = {0x8000, 0x8000, 0x8000, 0x8000},
       .expected = {{{0x80000000, 0x80000000, 0x80000000, 0x80000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x00000000, 0x00000000, 0x00000000, 0x00000000},
                     {0x80000000, 0x80000000, 0x80000000, 0x80000000}}}},
      {.name = "zero-weight NaN and infinity",
       .texels = {{{0x3c00, 0x8000, 0x3c00, 0x8000},
                   {0x7e01, 0x8000, 0x7c00, 0x8000},
                   {0x4200, 0x8000, 0x4200, 0x8000},
                   {0x7c00, 0x8000, 0xfc00, 0x8000}}},
       .mip = {0x2c01, 0xbc01, 0x7801, 0x1001},
       .expected = {{{0x3f800000, 0x80000000, 0x3f800000, 0x80000000},
                     {0x3fc00000, 0x00000000, 0x3fc00000, 0x00000000},
                     {0xffc00000, 0x00000000, 0xffc00000, 0x00000000},
                     {0x3f080200, 0xbf002000, 0x46802100, 0x39802000},
                     {0xffc00000, 0xbf002000, 0xffc00000, 0x39802000},
                     {0x3d802000, 0xbf802000, 0x47002000, 0x3a002000}}}},
  };
  constexpr struct Sample {
    const char *name;
    uint32_t index;
  } samples[] = {{"texel center", 0},
                 {"two contributing texels", 4},
                 {"four contributing texels", 5},
                 {"mip blend at texel centers", 32768},
                 {"mip and bilinear blend", 32773},
                 {"clamped mip endpoint", 65520}};
  for (uint32_t components : {1u, 2u, 4u}) {
    for (uint32_t pattern = 0; pattern < std::size(cases); ++pattern) {
      SCOPED_TRACE(components);
      const auto &test = cases[pattern];
      SCOPED_TRACE(test.name);
      const uint64_t base = 0x100000 + components * 0x100000 + pattern * 0x10000;
      const uint32_t bytes = components * 2;
      for (uint32_t level = 0; level < 2; ++level) {
        const auto mip = amdgpu::image_mip_layout(gfx12, 0, bytes, 2, 2, 2, level);
        ASSERT_TRUE(mip);
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x) {
            const auto address =
                gfx12 ? amdgpu::gfx12_image_address(base + mip->offset, x, y, mip->pitch, bytes, 0)
                      : amdgpu::gfx11_image_address(base + mip->offset, x, y, mip->pitch, bytes, 0);
            ASSERT_TRUE(address);
            const auto &texel = level ? test.mip : test.texels[y * 2 + x];
            ASSERT_EQ(access_->write(*address, std::as_bytes(std::span(texel).first(components))),
                      amdgpu::VmAccessOutcome::Complete);
          }
      }
      const uint32_t format = components == 1 ? 13 : components == 2 ? 29 : 57;
      // Missing Vulkan components are supplied by the descriptor selectors.
      const uint32_t selectors = components == 1 ? 0x204 : components == 2 ? 0x22c : 0xfac;
      const std::array<uint32_t, 8> descriptor{uint32_t(base >> 8),
                                               (format << (gfx12 ? 17 : 20)) | (1u << 30) |
                                                   (1u << (gfx12 ? 12 : 16)),
                                               1u << 14,
                                               (9u << 28) | selectors | (1u << (gfx12 ? 15 : 16)),
                                               0,
                                               0,
                                               0,
                                               0};
      for (uint32_t r = 0; r < descriptor.size(); ++r)
        wave_->debug_write_sgpr(8 + r, descriptor[r]);
      wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
      wave_->debug_write_sgpr(5, 256u << (gfx12 ? 13 : 12));
      wave_->debug_write_sgpr(6, (1 << 20) | (1 << 22) | (2 << 26));
      wave_->debug_write_sgpr(7, 0);
      wave_->set_exec(5);
      for (uint32_t sample = 0; sample < std::size(samples); ++sample) {
        SCOPED_TRACE(samples[sample].name);
        const uint32_t index = samples[sample].index;
        const float u = 0.25f + (index & 3u) / 8.0f;
        const float v = 0.25f + ((index >> 2) & 3u) / 8.0f;
        const float lod = (index >> 4) / 4096.0f;
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
          wave_->debug_write_vgpr(6, lane, std::bit_cast<uint32_t>(u));
          wave_->debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(v));
          wave_->debug_write_vgpr(4, lane, std::bit_cast<uint32_t>(lod));
          for (uint32_t c = 0; c < 4; ++c)
            wave_->debug_write_vgpr(8 + c, lane, 0xdeadbeef);
        }
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          const auto encoded = rdna4::build_vsample(29, {.dim = 1,
                                                         .dmask = 15,
                                                         .vdata = 8,
                                                         .rsrc = 8,
                                                         .samp = 4,
                                                         .vaddr0 = 6,
                                                         .vaddr1 = 0,
                                                         .vaddr2 = 4});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        } else {
          const auto encoded = rdna3::build_mimg(
              29,
              {.nsa = 1, .dim = 1, .dmask = 15, .vaddr = 6, .vdata = 8, .srsrc = 2, .ssamp = 1});
          std::copy(encoded.begin(), encoded.end(), words.begin());
          words[2] = 4 << 8;
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        ASSERT_NE(instruction->data(), nullptr);
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        for (uint32_t c = 0; c < 4; ++c) {
          const uint32_t expected = c < components ? test.expected[sample][c]
                                    : c == 3       ? 0x3f800000u
                                                   : 0;
          for (uint32_t lane : {0u, 2u})
            EXPECT_EQ(wave_->debug_read_vgpr(8 + c, lane), expected) << c;
          EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 1), 0xdeadbeefu);
        }
      }
    }
  }
}

TEST_P(GraphicsExportTest, NearestSamplingAndImageLoadsUseDistinctFloatRules) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t kR16Float = 13, kRg16Float = 29, kRgba16Float = 57;
  constexpr uint32_t kR32Float = 22, kRg32Float = 50, kRgba32Float = 63;
  constexpr uint32_t kR32Uint = 20, kRg32Uint = 48, kRgba32Uint = 61;
  constexpr uint32_t kImageLoad = 0, kImageSampleLz = 31;
  struct Case {
    const char *name;
    uint32_t format;
    std::array<uint32_t, 4> raw;
    std::array<uint32_t, 4> loaded;
    std::array<uint32_t, 4> sampled;
  };
  // Actual image-load and nearest-sample results from both physical RDNA3/4
  // cards, including half NaN payloads and single-precision subnormals.
  constexpr Case cases[] = {
      {.name = "FP16 zero and finite",
       .format = kRgba16Float,
       .raw = {0x00000000u, 0x00000089u, 0x00000112u, 0x0000019bu},
       .loaded = {0x00000000u, 0x37090000u, 0x37890000u, 0x37cd8000u},
       .sampled = {0x00000000u, 0x37090000u, 0x37890000u, 0x37cd8000u}},
      {.name = "FP16 positive subnormal",
       .format = kRgba16Float,
       .raw = {0x00000001u, 0x0000008au, 0x00000113u, 0x0000019cu},
       .loaded = {0x33800000u, 0x370a0000u, 0x37898000u, 0x37ce0000u},
       .sampled = {0x33800000u, 0x370a0000u, 0x37898000u, 0x37ce0000u}},
      {.name = "FP16 signaling NaN",
       .format = kRgba16Float,
       .raw = {0x00007c01u, 0x00007c8au, 0x00007d13u, 0x00007d9cu},
       .loaded = {0x7f802000u, 0x7f914000u, 0x7fa26000u, 0x7fb38000u},
       .sampled = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
      {.name = "FP16 quiet NaN",
       .format = kRgba16Float,
       .raw = {0x00007e00u, 0x00007e89u, 0x00007f12u, 0x00007f9bu},
       .loaded = {0x7fc00000u, 0x7fd12000u, 0x7fe24000u, 0x7ff36000u},
       .sampled = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
      {.name = "FP16 negative zero",
       .format = kRgba16Float,
       .raw = {0x00008000u, 0x00008089u, 0x00008112u, 0x0000819bu},
       .loaded = {0x80000000u, 0xb7090000u, 0xb7890000u, 0xb7cd8000u},
       .sampled = {0x80000000u, 0xb7090000u, 0xb7890000u, 0xb7cd8000u}},
      {.name = "FP16 negative signaling NaN",
       .format = kRgba16Float,
       .raw = {0x0000fc01u, 0x0000fc8au, 0x0000fd13u, 0x0000fd9cu},
       .loaded = {0xff802000u, 0xff914000u, 0xffa26000u, 0xffb38000u},
       .sampled = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
      {.name = "FP32 zero and finite",
       .format = kRgba32Float,
       .raw = {0x00000000u, 0x00890000u, 0x01120000u, 0x019b0000u},
       .loaded = {0x00000000u, 0x00890000u, 0x01120000u, 0x019b0000u},
       .sampled = {0x00000000u, 0x00890000u, 0x01120000u, 0x019b0000u}},
      {.name = "FP32 positive subnormal",
       .format = kRgba32Float,
       .raw = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u},
       .loaded = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u},
       .sampled = {0x00000000u, 0x008a0049u, 0x01130049u, 0x019c0049u}},
      {.name = "FP32 signaling NaN",
       .format = kRgba32Float,
       .raw = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u},
       .loaded = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u},
       .sampled = {0xffc00000u, 0x80000000u, 0x80925b80u, 0x811b5b80u}},
      {.name = "FP32 quiet NaN",
       .format = kRgba32Float,
       .raw = {0x7fc06dc0u, 0x80496dc0u, 0x80d26dc0u, 0x815b6dc0u},
       .loaded = {0x7fc06dc0u, 0x80496dc0u, 0x80d26dc0u, 0x815b6dc0u},
       .sampled = {0xffc00000u, 0x80000000u, 0x80d26dc0u, 0x815b6dc0u}},
      {.name = "FP32 negative subnormal",
       .format = kRgba32Float,
       .raw = {0x80008000u, 0x80898000u, 0x81128000u, 0x819b8000u},
       .loaded = {0x80008000u, 0x80898000u, 0x81128000u, 0x819b8000u},
       .sampled = {0x80000000u, 0x80898000u, 0x81128000u, 0x819b8000u}},
      {.name = "FP32 negative subnormal payload",
       .format = kRgba32Float,
       .raw = {0x80018049u, 0x808a8049u, 0x81138049u, 0x819c8049u},
       .loaded = {0x80018049u, 0x808a8049u, 0x81138049u, 0x819c8049u},
       .sampled = {0x80000000u, 0x808a8049u, 0x81138049u, 0x819c8049u}},
      {.name = "FP32 negative signaling NaN",
       .format = kRgba32Float,
       .raw = {0xff80db80u, 0x0009db80u, 0x0092db80u, 0x011bdb80u},
       .loaded = {0xff80db80u, 0x0009db80u, 0x0092db80u, 0x011bdb80u},
       .sampled = {0xffc00000u, 0x00000000u, 0x0092db80u, 0x011bdb80u}},
      {.name = "FP32 negative quiet NaN",
       .format = kRgba32Float,
       .raw = {0xffc0edc0u, 0x0049edc0u, 0x00d2edc0u, 0x015bedc0u},
       .loaded = {0xffc0edc0u, 0x0049edc0u, 0x00d2edc0u, 0x015bedc0u},
       .sampled = {0xffc00000u, 0x00000000u, 0x00d2edc0u, 0x015bedc0u}},
      {.name = "UINT subnormal bits",
       .format = kRgba32Uint,
       .raw = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u},
       .loaded = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u},
       .sampled = {0x00010049u, 0x008a0049u, 0x01130049u, 0x019c0049u}},
      {.name = "UINT NaN bits",
       .format = kRgba32Uint,
       .raw = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u},
       .loaded = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u},
       .sampled = {0x7f805b80u, 0x80095b80u, 0x80925b80u, 0x811b5b80u}},
      {.name = "FP32 normal boundary",
       .format = kRgba32Float,
       .raw = {0x007fffffu, 0x00800000u, 0x807fffffu, 0x80800000u},
       .loaded = {0x007fffffu, 0x00800000u, 0x807fffffu, 0x80800000u},
       .sampled = {0x00000000u, 0x00800000u, 0x80000000u, 0x80800000u}},
      {.name = "FP32 infinities and signed zero",
       .format = kRgba32Float,
       .raw = {0x7f800000u, 0xff800000u, 0x00000000u, 0x80000000u},
       .loaded = {0x7f800000u, 0xff800000u, 0x00000000u, 0x80000000u},
       .sampled = {0x7f800000u, 0xff800000u, 0x00000000u, 0x80000000u}},
      {.name = "FP32 extreme NaN payloads",
       .format = kRgba32Float,
       .raw = {0x7f800001u, 0xff800001u, 0x7fffffffu, 0xffffffffu},
       .loaded = {0x7f800001u, 0xff800001u, 0x7fffffffu, 0xffffffffu},
       .sampled = {0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u}},
  };
  constexpr uint64_t base = 0x180000;
  wave_->set_exec(5);
  // Sampling flushes FP32 texel subnormals even when VALU preserves them.
  wave_->set_mode_raw(0xf0);
  for (uint32_t components : {1u, 2u, 4u}) {
    SCOPED_TRACE(components);
    for (const auto &test : cases) {
      SCOPED_TRACE(test.name);
      const bool fp16 = test.format == kRgba16Float;
      std::array<uint16_t, 4> halves{};
      for (uint32_t c = 0; c < 4; ++c)
        halves[c] = static_cast<uint16_t>(test.raw[c]);
      ASSERT_EQ(
          access_->write(
              base, fp16 ? std::as_bytes(std::span<const uint16_t>(halves).first(components))
                         : std::as_bytes(std::span<const uint32_t>(test.raw).first(components))),
          amdgpu::VmAccessOutcome::Complete);
      cache_.invalidate_all();
      cu_->l1_vector().invalidate_all();
      uint32_t format;
      if (fp16)
        format = components == 1 ? kR16Float : components == 2 ? kRg16Float : kRgba16Float;
      else if (test.format == kRgba32Float)
        format = components == 1 ? kR32Float : components == 2 ? kRg32Float : kRgba32Float;
      else
        format = components == 1 ? kR32Uint : components == 2 ? kRg32Uint : kRgba32Uint;
      // Missing Vulkan components are supplied by the descriptor selectors.
      const uint32_t selectors = components == 1 ? 0x204 : components == 2 ? 0x22c : 0xfac;
      const std::array<uint32_t, 8> descriptor{
          uint32_t(base >> 8), format << (gfx12 ? 17 : 20), 0, (9u << 28) | selectors, 0, 0, 0, 0};
      for (uint32_t r = 0; r < descriptor.size(); ++r)
        wave_->debug_write_sgpr(8 + r, descriptor[r]);
      wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
      for (uint32_t r = 5; r < 8; ++r)
        wave_->debug_write_sgpr(r, 0);
      for (bool sample : {false, true}) {
        SCOPED_TRACE(sample ? "IMAGE_SAMPLE_LZ" : "IMAGE_LOAD");
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
          wave_->debug_write_vgpr(0, lane, sample ? std::bit_cast<uint32_t>(0.5f) : 0);
          wave_->debug_write_vgpr(1, lane, sample ? std::bit_cast<uint32_t>(0.5f) : 0);
          for (uint32_t c = 0; c < 4; ++c)
            wave_->debug_write_vgpr(8 + c, lane, 0xdeadbeef);
        }
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          if (sample) {
            const auto encoded = rdna4::build_vsample(kImageSampleLz, {.dim = 1,
                                                                       .dmask = 15,
                                                                       .vdata = 8,
                                                                       .rsrc = 8,
                                                                       .samp = 4,
                                                                       .vaddr0 = 0,
                                                                       .vaddr1 = 1});
            std::copy(encoded.begin(), encoded.end(), words.begin());
          } else {
            const auto encoded = rdna4::build_vimage(
                kImageLoad,
                {.dim = 1, .dmask = 15, .vdata = 8, .rsrc = 8, .vaddr0 = 0, .vaddr1 = 1});
            std::copy(encoded.begin(), encoded.end(), words.begin());
          }
        } else {
          const auto encoded = rdna3::build_mimg(
              sample ? kImageSampleLz : kImageLoad,
              {.dim = 1, .dmask = 15, .vaddr = 0, .vdata = 8, .srsrc = 2, .ssamp = 1});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        ASSERT_NE(instruction->data(), nullptr);
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        std::array<uint32_t, 4> expected = sample ? test.sampled : test.loaded;
        for (uint32_t c = components; c < 4; ++c)
          expected[c] = c == 3 ? (test.format == kRgba32Uint ? 1u : 0x3f800000u) : 0;
        for (uint32_t c = 0; c < 4; ++c) {
          for (uint32_t lane : {0u, 2u})
            EXPECT_EQ(wave_->debug_read_vgpr(8 + c, lane), expected[c]) << c;
          EXPECT_EQ(wave_->debug_read_vgpr(8 + c, 1), 0xdeadbeefu);
        }
      }
    }
  }
}

TEST_P(GraphicsExportTest, SampleLodUsesMipViewsDerivativesAndBias) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint32_t colors[] = {0xff000000, 0xff0000ff, 0xff00ff00, 0xffff0000};
  struct Case {
    const char *name;
    uint32_t opcode, mip_filter, first_level = 0;
    float lod, min_lod = 0, max_lod = 3, coordinate_step = 0;
    std::array<float, 4> expected;
    bool a16 = false;
    std::optional<std::array<float, 4>> gradient = std::nullopt;
    uint32_t width = 8, height = 8;
    uint32_t perf_mip = 0, perf_mod = 0;
    float sampler_bias = 0;
    int32_t secondary_bias = 0;
    uint32_t aniso_ratio = 0;
    float diagonal_offset = 0;
  };
  // Physical GFX11/12 mip weights, including a norm carry into the logarithm.
  const std::array rotated_gradient{0x1.f6cp-3f, -0x1.1ap-3f, -0x1.d48p-3f, 0x1.338p-4f};
  const std::array orthogonal_gradient{0.125f, 0.00250244140625f, -0.00250244140625f, 0.125f};
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const Case cases[] = {
      {.name = "NaN LOD before sampler bias",
       .opcode = 29,
       .mip_filter = 1,
       .lod = nan,
       .expected = {1, 0, 0, 1},
       .sampler_bias = 0.5f},
      {"positive infinite LOD", 29, 1, 0, infinity, 0, 3, 0, {0, 0, 1, 1}},
      {"negative infinite LOD", 29, 1, 0, -infinity, 0, 3, 0, {0, 0, 0, 1}},
      {"NaN quad derivatives", 27, 1, 0, 0, 0, 3, nan, {0, 0, 0, 1}},
      {"explicit nearest", 29, 1, 0, 1.25f, 0, 3, 0, {1, 0, 0, 1}},
      {"explicit rounded up", 29, 1, 0, 1.75f, 0, 3, 0, {0, 1, 0, 1}},
      {"below nearest threshold", 29, 1, 0, 0.498046875f - 1.0f / 8192, 0, 3, 0, {0, 0, 0, 1}},
      {"rounded nearest threshold", 29, 1, 0, 0.498046875f, 0, 3, 0, {1, 0, 0, 1}},
      {"mip interpolation", 29, 2, 0, 1.5f, 0, 3, 0, {0.5f, 0.5f, 0, 1}},
      {"minimum LOD", 29, 1, 0, 0, 2, 3, 0, {0, 1, 0, 1}},
      {"maximum LOD", 29, 1, 0, 3, 0, 1, 0, {1, 0, 0, 1}},
      {"view base", 29, 1, 1, 0, 0, 3, 0, {1, 0, 0, 1}},
      {"view upper bound", 29, 1, 1, 3, 0, 3, 0, {0, 0, 1, 1}},
      {"explicit derivatives", 28, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}},
      {"half derivatives", 57, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}},
      {"packed coordinates with half derivatives", 57, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}, true},
      {"implicit derivatives", 27, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}},
      // Hardware keeps coarse derivatives when the fourth lane's coordinates
      // are not affine. Fine derivatives would select the final blue mip.
      {.name = "coarse implicit derivatives",
       .opcode = 27,
       .mip_filter = 1,
       .lod = 0,
       .coordinate_step = 0.25f,
       .expected = {1, 0, 0, 1},
       .diagonal_offset = 2},
      {.name = "coarse packed implicit derivatives",
       .opcode = 27,
       .mip_filter = 1,
       .lod = 0,
       .coordinate_step = 0.25f,
       .expected = {1, 0, 0, 1},
       .a16 = true,
       .diagonal_offset = 2},
      {.name = "coarse biased derivatives",
       .opcode = 30,
       .mip_filter = 1,
       .lod = 1,
       .coordinate_step = 0.25f,
       .expected = {0, 1, 0, 1},
       .diagonal_offset = 2},
      {"shader bias", 30, 1, 0, 1, 0, 3, 0.25f, {0, 1, 0, 1}},
      {"forced zero", 31, 1, 0, 0, 0, 3, 0.5f, {0, 0, 0, 1}},
      {"packed explicit LOD", 29, 2, 0, 1.5f, 0, 3, 0, {0.5f, 0.5f, 0, 1}, true},
      {"packed coordinates with full derivatives", 28, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}, true},
      {"packed implicit coordinates", 27, 1, 0, 0, 0, 3, 0.5f, {0, 1, 0, 1}, true},
      {"packed coordinates with full bias", 30, 1, 0, 1, 0, 3, 0.25f, {0, 1, 0, 1}, true},
      {"packed forced zero", 31, 1, 0, 0, 0, 3, 0.5f, {0, 0, 0, 1}, true},
      {"axis logarithm", 28, 2, 0, 0, 0, 3, 0x1.1e3cb6p-1f, {0, 0.8359375f, 0.1640625f, 1}},
      {"correlated derivatives",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.83984375f, 0.16015625f, 0, 1},
       false,
       std::array{0.1875f, 0.0625f, 0.1875f, 0.0625f}},
      {"rotated derivatives",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.4375f, 0.5625f, 0, 1},
       false,
       rotated_gradient},
      {"rotated half derivatives",
       57,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.4375f, 0.5625f, 0, 1},
       false,
       rotated_gradient},
      {"rotated implicit derivatives",
       27,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.4375f, 0.5625f, 0, 1},
       false,
       rotated_gradient},
      {"coordinate alignment", 28, 2, 0, 0, 0, 3, 0, {0, 0, 0, 1}, false, orthogonal_gradient},
      {"half coordinate alignment", 57, 2, 0, 0, 0, 3, 0, {0, 0, 0, 1}, false, orthogonal_gradient},
      {"implicit coordinate alignment",
       27,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0, 0, 1},
       false,
       orthogonal_gradient},
      {"swapped coordinate exponents",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0.53125f, 0.46875f, 1},
       false,
       std::array{0x1.90ef84p-4f, 0x1.30801p-1f, 0x1.30801p-1f, 0x1.90ef84p-4f}},
      {"negative coordinate alignment",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.2734375f, 0.7265625f, 0, 1},
       false,
       std::array{-0.1253662109375f, -0.25f, -0.235595703125f, -0.2083740234375f}},
      {"non-power-of-two extents",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0.70703125f, 0.29296875f, 1},
       false,
       std::array{-0x1.7b0788p-2f, 0x1.8f028cp-5f, -0x1.2460ecp-4f, -0x1.87502ap-2f},
       13,
       9},
      {"extent multiplication carry",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0.3828125f, 0.6171875f, 1},
       false,
       std::array{0x1.47e36ep-5f, 0x1.91f32ep-2f, 0x1.c3679p-2f, 0x1.bc721p-3f},
       7,
       13},
      {"extent mantissa truncation",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0, 0.96875f, 0.03125f, 1},
       false,
       std::array{-0x1.119a6cp-13f, -0x1.35fd0ap-2f, -0x1.044334p-9f, 0x1.72db22p-5f},
       2051,
       9},
      {"zero coordinate exponent",
       28,
       2,
       0,
       0,
       0,
       3,
       0,
       {0.2265625f, 0.7734375f, 0, 1},
       false,
       std::array{0x1.5b3cd2p-10f, 0.0f, -0x1.0a3806p-10f, 0.0f},
       2051,
       9},
      {.name = "mip lower snap",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.25f,
       .expected = {1, 0, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4},
      {.name = "mip upper snap",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.75f,
       .expected = {0, 1, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4},
      {.name = "mip fraction truncation",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.49609375f,
       .expected = {0.5078125f, 0.4921875f, 0, 1},
       .perf_mip = 7,
       .perf_mod = 4},
      {.name = "maximum mip gain",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.50390625f,
       .expected = {0.453125f, 0.546875f, 0, 1},
       .perf_mip = 15,
       .perf_mod = 7},
      {.name = "disabled mip modulation",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.50390625f,
       .expected = {0.49609375f, 0.50390625f, 0, 1},
       .perf_mip = 15,
       .perf_mod = 0},
      {.name = "explicit rounding before bias",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1.001953125f,
       .expected = {0.99609375f, 0.00390625f, 0, 1},
       .sampler_bias = 0.00390625f},
      {.name = "explicit sampler bias",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1,
       .expected = {0.75f, 0.25f, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = 0.375f},
      {.name = "negative explicit sampler bias",
       .opcode = 29,
       .mip_filter = 2,
       .lod = 1,
       .expected = {0.75f, 0, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = -0.375f},
      {.name = "zero sampler bias",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .expected = {0.25f, 0, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = 0.375f},
      {.name = "bias bound before mip gain",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .min_lod = 0.375f,
       .expected = {0.25f, 0, 0, 1},
       .perf_mip = 10,
       .perf_mod = 4,
       .sampler_bias = -0.375f},
      {.name = "nearest mip ignores gain",
       .opcode = 29,
       .mip_filter = 1,
       .lod = 1,
       .expected = {0, 1, 0, 1},
       .perf_mip = 15,
       .perf_mod = 7,
       .sampler_bias = 0.625f},
      {.name = "negative secondary bias floor",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .expected = {0.83203125f, 0.16796875f, 0, 1},
       .perf_mip = 10,
       .perf_mod = 1,
       .sampler_bias = 1.375f,
       .secondary_bias = -17},
      {.name = "positive secondary bias floor",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .expected = {0.109375f, 0.890625f, 0, 1},
       .perf_mip = 10,
       .perf_mod = 2,
       .sampler_bias = 1.375f,
       .secondary_bias = 23},
      {.name = "disabled secondary modulation",
       .opcode = 31,
       .mip_filter = 2,
       .lod = 0,
       .expected = {0.625f, 0.375f, 0, 1},
       .perf_mip = 10,
       .perf_mod = 0,
       .sampler_bias = 1.375f,
       .secondary_bias = 23},
      {.name = "negative minor-axis difference",
       .opcode = 28,
       .mip_filter = 2,
       .lod = 0,
       .expected = {0.49609375f, 0, 0, 1},
       .gradient = std::array{-0x1.15654ap-1f, 0x1.00d27ep-3f, 0x1.982c54p-3f, 0x1.24184ep-3f},
       .aniso_ratio = 2},
      {.name = "positive minor-axis difference",
       .opcode = 28,
       .mip_filter = 2,
       .lod = 0,
       .expected = {0.39453125f, 0.60546875f, 0, 1},
       .gradient = std::array{0x1.ea1552p-1f, -0x1.0b398ap-3f, -0x1.ad88b6p-1f, 0x1.53ed8ep-1f},
       .aniso_ratio = 2},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    for (uint32_t level = 0; level < 4; ++level) {
      const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, test.width, test.height, 4, level);
      ASSERT_TRUE(mip);
      for (uint32_t y = 0; y < mip->height; ++y)
        for (uint32_t x = 0; x < mip->width; ++x) {
          const auto address =
              gfx12 ? amdgpu::gfx12_image_address(0x100000 + mip->offset, x, y, mip->pitch, 4, 0)
                    : amdgpu::gfx11_image_address(0x100000 + mip->offset, x, y, mip->pitch, 4, 0);
          ASSERT_TRUE(address);
          memory_.write32(*address, colors[level]);
        }
    }
    const std::array<uint32_t, 8> descriptor{
        0x1000,
        (42u << (gfx12 ? 17 : 20)) | (((test.width - 1) & 3u) << 30) | (3u << (gfx12 ? 12 : 16)) |
            (gfx12 ? test.first_level << 25 : 0),
        ((test.width - 1) >> 2) | ((test.height - 1) << 14),
        (9u << 28) | 0xfac | (3u << (gfx12 ? 15 : 16)) | (gfx12 ? 0 : test.first_level << 12),
        0,
        test.perf_mod << 20,
        0,
        0};
    for (uint32_t r = 0; r < descriptor.size(); ++r)
      wave_->debug_write_sgpr(8 + r, descriptor[r]);
    wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6) | (test.aniso_ratio << 9));
    wave_->debug_write_sgpr(5, uint32_t(test.min_lod * 256) |
                                   (uint32_t(test.max_lod * 256) << (gfx12 ? 13 : 12)) |
                                   (gfx12 ? 0 : test.perf_mip << 24));
    wave_->debug_write_sgpr(6, (test.mip_filter << 26) | (test.aniso_ratio ? 15u << 20 : 0) |
                                   (gfx12 ? (test.perf_mip & 3) << 30 : 0) |
                                   (uint32_t(int32_t(test.sampler_bias * 256)) & 0x3fff) |
                                   ((uint32_t(test.secondary_bias) & 63) << 14));
    wave_->debug_write_sgpr(7, gfx12 ? test.perf_mip >> 2 : 0);
    wave_->set_exec(9);
    const auto gradient =
        test.gradient.value_or(std::array{test.coordinate_step, 0.0f, 0.0f, test.coordinate_step});
    // Populate inactive quad lanes as well: implicit derivatives consume them.
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      const float offset = (lane & 3) == 3 ? test.diagonal_offset : 0;
      const float u = 0.25f + (lane & 1) * gradient[0] + ((lane >> 1) & 1) * gradient[2] + offset;
      const float v = 0.25f + (lane & 1) * gradient[1] + ((lane >> 1) & 1) * gradient[3] + offset;
      const std::array<uint32_t, 6> address_regs{6, 0, 4, 8, 9, 10};
      std::array<float, 6> values{u, v, test.lod};
      if (test.opcode == 28)
        values = {gradient[0], gradient[1], gradient[2], gradient[3], u, v};
      if (test.opcode == 57)
        values = {0, 0, u, v};
      if (test.opcode == 30)
        values = {test.lod, u, v};
      for (uint32_t r = 0; r < values.size(); ++r)
        wave_->debug_write_vgpr(address_regs[r], lane, std::bit_cast<uint32_t>(values[r]));
      if (test.opcode == 57) {
        wave_->debug_write_vgpr(address_regs[0], lane,
                                util::f32_to_f16(gradient[0]) |
                                    (uint32_t(util::f32_to_f16(gradient[1])) << 16));
        wave_->debug_write_vgpr(address_regs[1], lane,
                                util::f32_to_f16(gradient[2]) |
                                    (uint32_t(util::f32_to_f16(gradient[3])) << 16));
      }
      if (test.a16) {
        if (test.opcode == 30)
          wave_->debug_write_vgpr(address_regs[0], lane, util::f32_to_f16(test.lod));
        const uint32_t prefix = test.opcode == 28   ? 4
                                : test.opcode == 57 ? 2
                                : test.opcode == 30 ? 1
                                                    : 0;
        wave_->debug_write_vgpr(address_regs[prefix], lane,
                                util::f32_to_f16(values[prefix]) |
                                    (uint32_t(util::f32_to_f16(values[prefix + 1])) << 16));
        if (test.opcode == 29)
          wave_->debug_write_vgpr(address_regs[1], lane, util::f32_to_f16(test.lod));
      }
    }
    std::array<uint32_t, 4> words{};
    const uint8_t second_address = test.a16 && (test.opcode == 27 || test.opcode == 31) ? 255 : 0;
    const uint8_t fourth_address = test.a16 && test.opcode == 57 ? 255 : 8;
    if (gfx12) {
      const auto encoded = rdna4::build_vsample(test.opcode, {.dim = 1,
                                                              .a16 = test.a16,
                                                              .dmask = 15,
                                                              .vdata = 6,
                                                              .rsrc = 8,
                                                              .samp = 4,
                                                              .vaddr0 = 6,
                                                              .vaddr1 = second_address,
                                                              .vaddr2 = 4,
                                                              .vaddr3 = fourth_address});
      std::copy(encoded.begin(), encoded.end(), words.begin());
    } else {
      const auto encoded = rdna3::build_mimg(test.opcode, {.nsa = 1,
                                                           .dim = 1,
                                                           .dmask = 15,
                                                           .a16 = test.a16,
                                                           .vaddr = 6,
                                                           .vdata = 6,
                                                           .srsrc = 2,
                                                           .ssamp = 1});
      std::copy(encoded.begin(), encoded.end(), words.begin());
      words[2] = second_address | (4 << 8) | (uint32_t(fourth_address) << 16) | (9 << 24);
    }
    auto decoded = decoder_->decode(words.data());
    ASSERT_FALSE(decoded.failed());
    auto instruction = std::move(decoded).value();
    ASSERT_TRUE(instruction->is_memory_op());
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    ASSERT_FALSE(wave_->instruction_execution_failed());
    ASSERT_NE(instruction->data(), nullptr);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
    pipeline.issue(instruction.release(), *wave_);
    for (uint32_t lane : {0u, 3u})
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(6 + c, lane), std::bit_cast<uint32_t>(test.expected[c]))
            << lane << "," << c;
  }
}

TEST_P(GraphicsExportTest, SampleModifiersAndGatherMatchPhysicalTexels) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  // Physical GFX1100 and GFX1201 agree for this nonuniform three-mip image.
  // Cases distinguish offset signs, the CL minimum, every comparison function,
  // pre-filter comparisons, gather order and repeat/edge/border addressing.
  struct Case {
    const char *name;
    uint32_t opcode, test, lane;
    std::array<uint32_t, 4> expected;
    uint32_t resident_min = 0;
  };
  const Case cases[] = {
      {"sample_l_o", 39, 0, 0, {0x3f300000u, 0x3f300000u, 0x3f300000u, 0x3f300000u}},
      {"sample_l_o", 39, 3, 1, {0x3f960000u, 0x3f960000u, 0x3f960000u, 0x3f960000u}},
      {"sample_l_o", 39, 11, 2, {0x3f980000u, 0x3f980000u, 0x3f980000u, 0x3f980000u}},
      {"sample_l_o", 39, 19, 0, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"sample_l_o", 39, 23, 3, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"sample_d_o", 38, 3, 3, {0x3fcff800u, 0x3fcff800u, 0x3fcff800u, 0x3fcff800u}},
      {"sample_d_o", 38, 7, 3, {0x3fcff800u, 0x3fcff800u, 0x3fcff800u, 0x3fcff800u}},
      {"sample_d_cl", 65, 3, 3, {0x3fd69800u, 0x3fd69800u, 0x3fd69800u, 0x3fd69800u}},
      {"sample_d_cl", 65, 7, 3, {0x40000000u, 0x40000000u, 0x40000000u, 0x40000000u}},
      {"sample_d_cl_o", 71, 3, 3, {0x3fcff800u, 0x3fcff800u, 0x3fcff800u, 0x3fcff800u}},
      {"sample_d_cl_o", 71, 7, 3, {0x40000000u, 0x40000000u, 0x40000000u, 0x40000000u}},
      {"sample_d_o_g16", 59, 3, 3, {0x3fcff800u, 0x3fcff800u, 0x3fcff800u, 0x3fcff800u}},
      {"sample_d_o_g16", 59, 7, 3, {0x3fcff800u, 0x3fcff800u, 0x3fcff800u, 0x3fcff800u}},
      {"sample_d_cl_g16", 95, 3, 3, {0x3fd69800u, 0x3fd69800u, 0x3fd69800u, 0x3fd69800u}},
      {"sample_d_cl_g16", 95, 7, 3, {0x40000000u, 0x40000000u, 0x40000000u, 0x40000000u}},
      {"sample_d_cl_o_g16", 85, 3, 3, {0x3fcff800u, 0x3fcff800u, 0x3fcff800u, 0x3fcff800u}},
      {"sample_d_cl_o_g16", 85, 7, 3, {0x40000000u, 0x40000000u, 0x40000000u, 0x40000000u}},
      {"sample_c_l", 34, 3, 12, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"sample_c_l", 34, 7, 12, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      {"sample_c_l", 34, 11, 12, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"sample_c_l", 34, 15, 12, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      {"sample_c_l", 34, 19, 12, {0x3eb00000u, 0x3eb00000u, 0x3eb00000u, 0x3eb00000u}},
      {"sample_c_l", 34, 23, 12, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      {"sample_c_l", 34, 27, 12, {0x3e000000u, 0x3e000000u, 0x3e000000u, 0x3e000000u}},
      {"sample_c_l", 34, 31, 12, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      {"sample_c_l_o", 44, 19, 4, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      {"sample_c_d_cl_o_g16", 86, 27, 12, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"gather4_l", 48, 3, 1, {0x3fa00000u, 0x3fa80000u, 0x3f880000u, 0x3f800000u}},
      {"gather4_l", 48, 19, 1, {0x3fa00000u, 0x3fa80000u, 0x3f880000u, 0x3f800000u}},
      {"gather4_l", 48, 23, 1, {0x40000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"gather4_lz_o", 54, 3, 1, {0x3f000000u, 0x3f100000u, 0x3ea00000u, 0x3e800000u}},
      {"gather4_lz_o", 54, 19, 1, {0x3f000000u, 0x3f100000u, 0x3ea00000u, 0x3e800000u}},
      {"gather4_lz_o", 54, 23, 1, {0x3e000000u, 0x3e400000u, 0x00000000u, 0x00000000u}},
      {"gather4_c_l", 99, 3, 1, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"gather4_c_l", 99, 19, 1, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"gather4_c_l", 99, 23, 1, {0x3f800000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"gather4_c_lz_o", 55, 3, 1, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"gather4_c_lz_o", 55, 19, 1, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {"gather4_c_lz_o", 55, 23, 1, {0x3f800000u, 0x3f800000u, 0x00000000u, 0x00000000u}},
      {"gather4h", 144, 3, 1, {0x3f800000u, 0x3f800000u, 0x3f880000u, 0x3f880000u}},
      {"gather4h", 144, 19, 1, {0x00000000u, 0x3f800000u, 0x3f880000u, 0x00000000u}},
      {"gather4h", 144, 23, 1, {0x00000000u, 0x3f800000u, 0x3f880000u, 0x00000000u}},
      {"sample_lz", 31, 7, 0, {0x3e591000u, 0x3e591000u, 0x3e591000u, 0x3e591000u}, 25},
      {"sample_l", 29, 8, 0, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}, 128},
      {"gather4_lz", 50, 15, 0, {0x3e800000u, 0x3ea00000u, 0x3d800000u, 0x00000000u}, 230},
      {"gather4_lz", 50, 16, 0, {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, 256},
      {"gather4_c_l", 99, 20, 12, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}, 281},
      {"sample_b", 30, 3, 0, {0x3f8cd400u, 0x3f8cd400u, 0x3f8cd400u, 0x3f8cd400u}},
      {"sample_b_cl", 66, 7, 0, {0x40000000u, 0x40000000u, 0x40000000u, 0x40000000u}},
      {"sample_c_b_cl_o", 75, 23, 12, {0x3f800000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
  };
  for (uint32_t level = 0; level < 3; ++level) {
    const uint32_t base = level == 0 ? 768 : level == 1 ? 256 : 0;
    for (uint32_t y = 0; y < (4u >> level); ++y)
      for (uint32_t x = 0; x < (4u >> level); ++x)
        memory_.write32(0x100000 + base + y * (gfx12 ? 128 : 256) + x * 4,
                        std::bit_cast<uint32_t>(float(level * 16 + y * 4 + x) / 16));
  }
  const std::array<uint32_t, 8> descriptor{
      0x1000,   (22u << (gfx12 ? 17 : 20)) | (3u << 30) | (2u << (gfx12 ? 12 : 16)),
      3u << 14, 0x90000fac | (2u << (gfx12 ? 15 : 16)),
      0,        0,
      0,        0};
  for (uint32_t i = 0; i < 8; ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  for (const auto &test : cases) {
    const std::string_view name = test.name;
    const bool bias = name.find("_b") != name.npos;
    const bool offset = name.find("_o") != name.npos;
    const bool compare = name.find("_c_") != name.npos;
    const bool gradient = name.find("_d") != name.npos;
    const bool g16 = name.ends_with("_g16");
    const bool clamp = name.find("_cl") != name.npos;
    const bool explicit_lod = name.ends_with("_l") || name.find("_l_o") != name.npos;
    const bool gather = name.starts_with("gather");
    uint32_t wrap = (test.test / 8) % 3;
    wrap = wrap == 2 ? 6 : wrap == 1 ? 2 : 0;
    const uint32_t linear = test.test & 1, mip = 1 + ((test.test >> 1) & 1);
    wave_->debug_write_sgpr(4, wrap | (wrap << 3) | (((test.test >> 2) & 7) << 12));
    wave_->debug_write_sgpr(5, 512u << (gfx12 ? 13 : 12));
    wave_->debug_write_sgpr(6, (linear << 20) | (linear << 22) | (mip << 26));
    wave_->debug_write_sgpr(7, 0);
    wave_->debug_write_sgpr(13, test.resident_min << (gfx12 ? 26 : 27));
    wave_->debug_write_sgpr(14, test.resident_min >> (gfx12 ? 6 : 5));
    const float lod = (test.test % 8) * 0.375f - 0.25f;
    for (bool a16 : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << name << ',' << test.test << ',' << test.lane << ',' << a16);
      wave_->set_exec(uint64_t{1} << test.lane);
      // Quad helper lanes are populated but must never receive a result.
      for (uint32_t lane = 0; lane < 32; ++lane) {
        std::vector<uint32_t> coordinates;
        if (offset)
          coordinates.push_back(test.test & 4 ? 0x3f01 : 0x013f);
        if (bias)
          coordinates.push_back(a16 ? util::f32_to_f16(0.5f) : std::bit_cast<uint32_t>(0.5f));
        if (compare)
          coordinates.push_back(std::bit_cast<uint32_t>((lane / 4) * 0.25f));
        if (gradient) {
          coordinates.push_back(g16 ? util::f32_to_f16(0.75f) : std::bit_cast<uint32_t>(0.75f));
          coordinates.push_back(0);
          if (!g16) {
            coordinates.push_back(0);
            coordinates.push_back(0);
          }
        }
        const float u = lane & 1 ? 0.5f : 0.125f, v = lane & 2 ? 0.625f : 0.25f;
        if (a16) {
          coordinates.push_back(util::f32_to_f16(u) | (uint32_t(util::f32_to_f16(v)) << 16));
          if (explicit_lod || clamp)
            coordinates.push_back(util::f32_to_f16(lod));
        } else {
          coordinates.push_back(std::bit_cast<uint32_t>(u));
          coordinates.push_back(std::bit_cast<uint32_t>(v));
          if (explicit_lod || clamp)
            coordinates.push_back(std::bit_cast<uint32_t>(lod));
        }
        for (uint32_t i = 0; i < coordinates.size(); ++i)
          wave_->debug_write_vgpr(i, lane, coordinates[i]);
        for (uint32_t i = 12; i < 16; ++i)
          wave_->debug_write_vgpr(i, lane, 0xdeadbeef);
      }
      std::array<uint32_t, 4> words{};
      if (gfx12) {
        const auto encoded = rdna4::build_vsample(test.opcode, {.dim = 1,
                                                                .a16 = a16,
                                                                .dmask = uint8_t(gather ? 1 : 15),
                                                                .vdata = 12,
                                                                .rsrc = 8,
                                                                .samp = 4,
                                                                .vaddr0 = 0,
                                                                .vaddr1 = 1,
                                                                .vaddr2 = 2,
                                                                .vaddr3 = 3});
        std::copy(encoded.begin(), encoded.end(), words.begin());
      } else {
        const auto encoded = rdna3::build_mimg(test.opcode, {.dim = 1,
                                                             .dmask = uint8_t(gather ? 1 : 15),
                                                             .a16 = a16,
                                                             .vaddr = 0,
                                                             .vdata = 12,
                                                             .srsrc = 2,
                                                             .ssamp = 1});
        std::copy(encoded.begin(), encoded.end(), words.begin());
      }
      auto decoded = decoder_->decode(words.data());
      ASSERT_FALSE(decoded.failed());
      auto instruction = std::move(decoded).value();
      ASSERT_TRUE(instruction->is_memory_op());
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      ASSERT_FALSE(wave_->instruction_execution_failed());
      ASSERT_NE(instruction->data(), nullptr);
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
      pipeline.issue(instruction.release(), *wave_);
      for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(wave_->debug_read_vgpr(12 + i, test.lane), test.expected[i]) << i;
        EXPECT_EQ(wave_->debug_read_vgpr(12 + i, test.lane ^ 1), 0xdeadbeefu);
      }
    }
  }
}

TEST_P(GraphicsExportTest, PackedGradientComponentsSelectMipIndependentlyOfCoordinates) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (bool cube : {false, true}) {
    // The non-square 2D image distinguishes U/V halves. Cube neighbors select
    // opposite faces, so implicit derivatives would select a different mip.
    const uint32_t height = cube ? 8 : 4, layers = cube ? 6 : 1;
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    for (uint32_t layer = 0; layer < layers; ++layer)
      for (uint32_t level = 0; level < 4; ++level) {
        const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, 8, height, 4, level);
        ASSERT_TRUE(mip);
        const uint64_t base = 0x100000 + layer * mip->slice_size + mip->offset;
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x) {
            const auto address = gfx12 ? amdgpu::gfx12_image_address(base, x, y, mip->pitch, 4, 0)
                                       : amdgpu::gfx11_image_address(base, x, y, mip->pitch, 4, 0);
            ASSERT_TRUE(address);
            memory_.write32(*address, 0xff000000 | (layer << 8) | level);
          }
      }
    const std::array<uint32_t, 8> descriptor{
        0x1000,
        (46u << (gfx12 ? 17 : 20)) | (3u << 30) | (3u << (gfx12 ? 12 : 16)),
        1 | ((height - 1) << 14),
        ((cube ? 11u : 9u) << 28) | 0xfac | (3u << (gfx12 ? 15 : 16)),
        layers - 1,
        0,
        0,
        0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
    wave_->debug_write_sgpr(5, 768u << (gfx12 ? 13 : 12));
    wave_->debug_write_sgpr(6, 1u << 26);
    wave_->debug_write_sgpr(7, 0);
    wave_->set_exec(15);
    for (bool a16 : {false, true}) {
      SCOPED_TRACE(testing::Message() << "cube=" << cube << ", a16=" << a16);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        // Each lane supplies exactly one nonzero explicit derivative. Body
        // coordinates are constant, so their quad differences cannot stand in.
        const uint32_t gradient = uint32_t(util::f32_to_f16(0.5f)) << (16 * (lane % 2));
        wave_->debug_write_vgpr(0, lane, lane < 2 ? gradient : 0);
        wave_->debug_write_vgpr(1, lane, lane >= 2 ? gradient : 0);
        const float coordinate = cube ? 1.25f : 0.25f;
        const uint32_t packed = util::f32_to_f16(coordinate);
        wave_->debug_write_vgpr(
            2, lane, a16 ? packed | (packed << 16) : std::bit_cast<uint32_t>(coordinate));
        wave_->debug_write_vgpr(
            3, lane, a16 ? util::f32_to_f16(float(lane)) : std::bit_cast<uint32_t>(coordinate));
        wave_->debug_write_vgpr(4, lane, std::bit_cast<uint32_t>(float(lane)));
      }
      wave_->debug_write_vgpr(12, 4, 0xdeadbeef);
      ASSERT_NO_FATAL_FAILURE(sample(57, cube ? 3 : 1, a16));
      for (uint32_t lane = 0; lane < 4; ++lane) {
        EXPECT_EQ(wave_->debug_read_vgpr(12, lane), cube || !(lane & 1) ? 2u : 1u) << lane;
        EXPECT_EQ(wave_->debug_read_vgpr(13, lane), cube ? lane : 0u) << lane;
      }
      EXPECT_EQ(wave_->debug_read_vgpr(12, 4), 0xdeadbeefu);
    }
  }
}

TEST_P(GraphicsExportTest, SampledArrayViewsClampLayersAndKeepMipCoordinatesSeparate) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  std::optional<uint64_t> (*image_address)(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                           uint32_t) =
      gfx12 ? amdgpu::gfx12_image_address : amdgpu::gfx11_image_address;
  for (bool one_d : {false, true}) {
    const uint32_t spatial = one_d ? 1 : 2;
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    for (uint32_t layer = 0; layer < 5; ++layer)
      for (uint32_t level = 0; level < 4; ++level) {
        const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, 8, one_d ? 1 : 8, 4, level);
        ASSERT_TRUE(mip);
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x)
            memory_.write32(*image_address(0x100000 + layer * mip->slice_size + mip->offset, x, y,
                                           mip->pitch, 4, 0),
                            0xff000000 | (layer << 8) | level);
      }
    for (uint32_t type : {one_d ? 8u : 9u, one_d ? 12u : 13u})
      for (bool a16 : {false, true})
        for (bool array : {false, true})
          for (const auto &[opcode, name] : {std::pair{27u, "IMAGE_SAMPLE"},
                                             {28u, "IMAGE_SAMPLE_D"},
                                             {57u, "IMAGE_SAMPLE_D_G16"},
                                             {29u, "IMAGE_SAMPLE_L"},
                                             {30u, "IMAGE_SAMPLE_B"},
                                             {31u, "IMAGE_SAMPLE_LZ"}}) {
            SCOPED_TRACE(testing::Message() << "type=" << type << ", a16=" << a16
                                            << ", array=" << array << ", opcode=" << name);
            const std::array<uint32_t, 8> descriptor{
                0x1000,
                (46u << (gfx12 ? 17 : 20)) | (3u << 30) | (3u << (gfx12 ? 12 : 16)),
                1 | (one_d ? 0 : 7u << 14),
                (type << 28) | 0xfac | (3u << (gfx12 ? 15 : 16)),
                type >= 12 ? (2u << 16) | 4 : 0,
                0,
                0,
                0};
            for (uint32_t r = 0; r < descriptor.size(); ++r)
              wave_->debug_write_sgpr(8 + r, descriptor[r]);
            wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
            wave_->debug_write_sgpr(5, 768u << (gfx12 ? 13 : 12));
            wave_->debug_write_sgpr(6, 1u << 26);
            wave_->debug_write_sgpr(7, 0);
            wave_->set_exec(15);
            const uint32_t prefix = opcode == 28   ? 2 * spatial
                                    : opcode == 57 ? 2
                                    : opcode == 30 ? 1
                                                   : 0;
            const std::array<uint32_t, 7> registers{6, 0, 4, 8, 9, 10, 11};
            for (uint32_t lane = 0; lane < 4; ++lane) {
              std::array<float, 7> values{};
              if (opcode == 28)
                values = {0.5f, 0, 0, 0.5f};
              if (opcode == 30)
                values[0] = 1;
              values[prefix] = (lane & 1) * 0.5f;
              if (!one_d)
                values[prefix + 1] = ((lane >> 1) & 1) * 0.5f;
              if (array)
                values[prefix + spatial] = lane == 0   ? -0.5f
                                           : lane == 1 ? 0.5f
                                           : lane == 2 ? 1.5f
                                                       : 100.0f;
              if (opcode == 29)
                values[prefix + spatial + array] = 1.0f;
              for (uint32_t i = 0; i < values.size(); ++i)
                wave_->debug_write_vgpr(registers[i], lane, std::bit_cast<uint32_t>(values[i]));
              if (opcode == 57) {
                wave_->debug_write_vgpr(registers[0], lane, util::f32_to_f16(0.5f));
                wave_->debug_write_vgpr(registers[1], lane, uint32_t(util::f32_to_f16(0.5f)) << 16);
              }
              if (a16) {
                if (opcode == 30)
                  wave_->debug_write_vgpr(registers[0], lane, util::f32_to_f16(values[0]));
                const uint32_t count = spatial + array + (opcode == 29);
                for (uint32_t i = 0; i < count; i += 2)
                  wave_->debug_write_vgpr(
                      registers[prefix + i / 2], lane,
                      util::f32_to_f16(values[prefix + i]) |
                          (uint32_t(util::f32_to_f16(i + 1 < count ? values[prefix + i + 1] : 0))
                           << 16));
              }
            }
            std::array<uint32_t, 4> words{};
            if (gfx12) {
              const auto encoded = rdna4::build_vsample(
                  opcode, {.dim = uint8_t(array ? (one_d ? 4 : 5) : (one_d ? 0 : 1)),
                           .a16 = a16,
                           .dmask = 15,
                           .vdata = 12,
                           .rsrc = 8,
                           .samp = 4,
                           .vaddr0 = 6,
                           .vaddr1 = 0,
                           .vaddr2 = 4,
                           .vaddr3 = 8});
              std::copy(encoded.begin(), encoded.end(), words.begin());
            } else {
              const auto encoded = rdna3::build_mimg(
                  opcode, {.nsa = 1,
                           .dim = uint8_t(array ? (one_d ? 4 : 5) : (one_d ? 0 : 1)),
                           .dmask = 15,
                           .a16 = a16,
                           .vaddr = 6,
                           .vdata = 12,
                           .srsrc = 2,
                           .ssamp = 1});
              std::copy(encoded.begin(), encoded.end(), words.begin());
              words[2] = (4 << 8) | (8 << 16) | (9 << 24);
            }
            auto decoded = decoder_->decode(words.data());
            ASSERT_FALSE(decoded.failed());
            auto instruction = std::move(decoded).value();
            ASSERT_TRUE(instruction->is_memory_op());
            ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
            ASSERT_FALSE(wave_->instruction_execution_failed());
            amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
            pipeline.issue(instruction.release(), *wave_);
            for (uint32_t lane = 0; lane < 4; ++lane) {
              const uint32_t level = opcode == 31 ? 0 : opcode == 29 ? 1 : opcode == 30 ? 3 : 2;
              const uint32_t layer = type < 12 ? 0 : !array || lane < 2 ? 2 : 4;
              EXPECT_EQ(wave_->debug_read_vgpr(12, lane), level);
              EXPECT_EQ(wave_->debug_read_vgpr(13, lane), layer);
              EXPECT_EQ(wave_->debug_read_vgpr(14, lane), 0u);
              EXPECT_EQ(wave_->debug_read_vgpr(15, lane), 255u);
            }
          }
  }
}

TEST_P(GraphicsExportTest, TrilinearUsesDistinctWeightsForNonuniformMipLevels) {
  for (bool translated : {false, true}) {
    SCOPED_TRACE(translated);
    if (translated) {
      const auto address_space =
          vm_.register_address_space(1, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                     std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_));
      ASSERT_TRUE(address_space);
      wave_->set_address_space(address_space);
    }
    for (bool srgb : {false, true}) {
      SCOPED_TRACE(srgb);
      // The host refills backing memory directly between image fixtures.
      cu_->l1_vector().invalidate_all();
      cache_.invalidate_all();
      const uint32_t width = srgb ? 7 : 13, height = srgb ? 5 : 9, levels = srgb ? 3 : 4;
      const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
      for (uint32_t level = 0; level < levels; ++level) {
        const auto mip = amdgpu::image_mip_layout(gfx12, 0, 4, width, height, levels, level);
        ASSERT_TRUE(mip);
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x) {
            const auto address =
                gfx12 ? amdgpu::gfx12_image_address(0x100000 + mip->offset, x, y, mip->pitch, 4, 0)
                      : amdgpu::gfx11_image_address(0x100000 + mip->offset, x, y, mip->pitch, 4, 0);
            ASSERT_TRUE(address);
            uint32_t color = (17 * x + 31 * y + 53 * level) |
                             ((13 * x + 47 * y + 19 * level) << 8) |
                             ((11 * x + 23 * y + 29 * level) << 16) | 0xff000000;
            if (!srgb) {
              color = (x + 13 * y + 117 * level) * 1664525u + 1013904223u;
              color ^= color >> 16;
              color *= 2246822519u;
            }
            memory_.write32(*address, color);
          }
      }
      const std::array<uint32_t, 8> descriptor{
          0x1000,
          ((srgb ? 66u : 42u) << (gfx12 ? 17 : 20)) | (((width - 1) & 3) << 30) |
              ((levels - 1) << (gfx12 ? 12 : 16)),
          ((width - 1) >> 2) | ((height - 1) << 14),
          (9u << 28) | 0xfac | ((levels - 1) << (gfx12 ? 15 : 16)),
          0,
          0,
          0,
          0};
      for (uint32_t r = 0; r < descriptor.size(); ++r)
        wave_->debug_write_sgpr(8 + r, descriptor[r]);
      wave_->debug_write_sgpr(4, 2 | (2 << 3) | (2 << 6));
      wave_->debug_write_sgpr(5, ((levels - 1) * 256) << (gfx12 ? 13 : 12));
      wave_->debug_write_sgpr(6, (1 << 20) | (1 << 22) | (2 << 26));
      wave_->debug_write_sgpr(7, 0);
      wave_->set_exec(1);
      struct Case {
        uint32_t index;
        std::array<uint32_t, 4> expected;
      };
      // Physical gfx1100 and gfx1201 results for UV=(index%256,index/256)/256,
      // LOD=(index%513)/256. All eight taps contribute at the interior coordinates.
      const std::array<Case, 5> srgb_cases{{
          {343, {0x3da6ca8a, 0x3c5f2280, 0x3cd123dc, 0x3f800000}},
          {6764, {0x3d122bb4, 0x3c6a7780, 0x3c6a0f81, 0x3f800000}},
          {18374, {0x3e011439, 0x3cb9adb7, 0x3d1f1fdd, 0x3f800000}},
          {32467, {0x3e557300, 0x3e40d053, 0x3db0d1b7, 0x3f800000}},
          {49141, {0x3e2372b6, 0x3d62df4e, 0x3d58d6c0, 0x3f800000}},
      }};
      // Independent hardware captures use pseudorandom RGBA texels and exercise
      // rounding of each weighted mip before the final texel-value rounding.
      const std::array<Case, 5> unorm_cases{{
          {239, {0x3e83a808, 0x3eecb969, 0x3f113f6f, 0x3e9cd8d9}},
          {612, {0x3e86c6a7, 0x3ed3f7c8, 0x3ef3c9fa, 0x3ec74959}},
          {1344, {0x3f0acd1d, 0x3ef5b707, 0x3f3414c5, 0x3f18ab03}},
          {1769, {0x3eda8d2d, 0x3ebf60f1, 0x3f24469f, 0x3effae4e}},
          {2099, {0x3ed0b6e7, 0x3ee7aacb, 0x3ed90a0a, 0x3f18b8c1}},
      }};
      for (const auto &test : srgb ? srgb_cases : unorm_cases) {
        SCOPED_TRACE(test.index);
        wave_->debug_write_vgpr(0, 0, std::bit_cast<uint32_t>((test.index % 256) / 256.0f));
        wave_->debug_write_vgpr(1, 0, std::bit_cast<uint32_t>((test.index / 256) / 256.0f));
        wave_->debug_write_vgpr(2, 0, std::bit_cast<uint32_t>((test.index % 513) / 256.0f));
        if (!srgb) {
          uint32_t seed = test.index * 1664525u + 1013904223u;
          seed ^= seed >> 16;
          seed *= 2246822519u;
          wave_->debug_write_vgpr(0, 0, std::bit_cast<uint32_t>((seed & 1023) / 512.0f - 0.5f));
          wave_->debug_write_vgpr(1, 0,
                                  std::bit_cast<uint32_t>(((seed >> 10) & 1023) / 512.0f - 0.5f));
          wave_->debug_write_vgpr(2, 0, std::bit_cast<uint32_t>(((seed >> 20) % 2049) / 512.0f));
        }
        std::array<uint32_t, 4> words{};
        if (gfx12) {
          const auto encoded = rdna4::build_vsample(29, {.dim = 1,
                                                         .dmask = 15,
                                                         .vdata = 4,
                                                         .rsrc = 8,
                                                         .samp = 4,
                                                         .vaddr0 = 0,
                                                         .vaddr1 = 1,
                                                         .vaddr2 = 2});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        } else {
          const auto encoded = rdna3::build_mimg(
              29, {.dim = 1, .dmask = 15, .vaddr = 0, .vdata = 4, .srsrc = 2, .ssamp = 1});
          std::copy(encoded.begin(), encoded.end(), words.begin());
        }
        auto decoded = decoder_->decode(words.data());
        ASSERT_FALSE(decoded.failed());
        auto instruction = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
        ASSERT_FALSE(wave_->instruction_execution_failed());
        ASSERT_NE(instruction->data(), nullptr);
        const auto *state = instruction->data_as<amdgpu::VectorMemState>();
        ASSERT_NE(state->image_sample, nullptr);
        EXPECT_EQ(state->image_sample->tap_count, 8u);
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &cache_);
        pipeline.issue(instruction.release(), *wave_);
        for (uint32_t c = 0; c < 4; ++c) {
          const auto actual = wave_->debug_read_vgpr(4 + c, 0);
          EXPECT_EQ(actual, test.expected[c]) << c;
        }
      }
    }
  }
}

TEST_P(GraphicsExportTest, StencilComparisonsOperationsAndMasksRespectDepthAndFacing) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr bool comparisons[]{false, true, false, true, false, true, false, true};
  constexpr uint8_t results[]{0x5a, 0,    255,  0x36, 0xc3, 0x5b, 0x59, 0xa5,
                              0x5b, 0x59, 0x42, 0xdb, 0x99, 0xbd, 0x24, 0x66};
  for (bool back : {false, true})
    for (uint32_t comparison = 0; comparison < 8; ++comparison)
      for (bool depth_pass : {false, true})
        for (uint32_t operation = 0; operation < 16; ++operation) {
          SCOPED_TRACE(testing::Message()
                       << back << "," << comparison << "," << depth_pass << "," << operation);
          auto state = rectangle_state();
          auto &ctx = state.context_registers;
          ctx[gfx12 ? 5 : 7] = (3 << 16) | 3;
          const uint32_t swizzle = gfx12 ? 3 : 24;
          ctx[gfx12 ? 6 : 0x10] = 3 | (swizzle << 4);
          ctx[gfx12 ? 7 : 0x11] = 1 | (swizzle << 4);
          const auto address = [&](uint32_t base, uint32_t x, uint32_t y, uint32_t bytes) {
            return *(gfx12 ? amdgpu::gfx12_image_address(base, x, y, 4, bytes, swizzle)
                           : amdgpu::gfx11_image_address(base, x, y, 4, bytes, swizzle));
          };
          ctx[gfx12 ? 8 : 0x12] = ctx[gfx12 ? 10 : 0x14] = 0x2000;
          ctx[gfx12 ? 0xc : 0x13] = ctx[gfx12 ? 0xe : 0x15] = 0x3000;
          ctx[gfx12 ? 0x1c : 0x200] =
              0x87 | (depth_pass ? 0x70 : 0) | (comparison << (back ? 20 : 8));
          // Deliberately poison the unused face and unused outcome operations.
          const uint32_t slot = (back ? 12 : 0) + (!comparisons[comparison] ? 0
                                                   : depth_pass             ? 4
                                                                            : 8);
          ctx[gfx12 ? 0x1d : 0x10b] = (0xffffffu & ~(15u << slot)) | (operation << slot);
          if (gfx12) {
            const uint32_t shift = back ? 8 : 0;
            ctx[0x22] = 0x36u << shift;
            ctx[0x23] = 0xc3u << shift;
            ctx[0x24] = 0x0fu << shift;
            ctx[0x25] = 0xf0u << shift;
          } else {
            ctx[back ? 0x10d : 0x10c] = 0xc3f00f36;
          }
          const bool shader_stencil = operation == 3;
          if (shader_stencil) {
            ctx[gfx12 ? 0x1b : 0x203] |= 2; // STENCIL_TEST_VAL_EXPORT_ENABLE.
            ctx[gfx12 ? 0x194 : 0x1c4] = 2; // SPI_SHADER_32_GR.
            ctx[gfx12 ? 0x22 : back ? 0x10d : 0x10c] ^= 0xf9u << (gfx12 && back ? 8 : 0);
          }
          ctx[gfx12 ? 0x207 : 0x205] = back ? 4 : 0;
          ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(2.0f);
          ctx[0x113] = std::bit_cast<uint32_t>(1.0f);
          ctx[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(1.0f);
          ctx[0x90] = 1 | (1 << 16);
          ctx[0x91] = (3 - gfx12) | ((3 - gfx12) << 16);
          for (uint32_t i = 0; i < 16; ++i) {
            memory_.write32(address(0x200000, i % 4, i / 4, 4), std::bit_cast<uint32_t>(1.0f));
            memory_.write_block(address(0x300000, i % 4, i / 4, 1), std::array<uint8_t, 1>{0x5a});
          }
          auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
          export_rectangle_vertices(*draw);
          ASSERT_TRUE(draw->advance(*access_));
          if (shader_stencil) {
            initialize_fragment(draw);
            for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
              draw->export_lane(*wave_, lane, 8, 2, {0, 0x12345636, 0, 0});
            ASSERT_FALSE(wave_->instruction_execution_failed());
          }
          EXPECT_FALSE(draw->advance(*access_));
          for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 4; ++x) {
              const bool covered = x >= 1 && x < 3 && y >= 1 && y < 3;
              const auto stencil = memory_.read32(address(0x300000, x, y, 1)) & 255;
              EXPECT_EQ(stencil, covered ? (results[operation] & 0xf0) | 0xau : 0x5au);
              EXPECT_EQ(memory_.read32(address(0x200000, x, y, 4)),
                        covered && comparisons[comparison] && depth_pass ? 0u : 0x3f800000u);
            }
        }
}

TEST_P(GraphicsExportTest, DepthClearAndComparisonsUseTiledD16AndD32) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    const char *name;
    float vertex_z = 0, scale = 1, offset = 0, minimum = 0, maximum = 1;
    bool disable_clamp = false;
    float expected = 0;
    uint16_t expected_d16 = 0;
    bool disable_samples = false;
    bool fragment_export = false;
    uint64_t export_mask = ~uint64_t{0};
    bool enable_stencil = false;
    uint32_t stencil_info = 0;
  };
  constexpr Case cases[] = {
      {"zero depth", 0, 1, 0, 0, 1, false, 0, 0},
      {.name = "inactive stencil surface", .enable_stencil = true, .stencil_info = 0x20100180},
      {"far depth clamped", 2, 0.5f, 0.25f, 0.25f, 0.75f, false, 0.75f, 49151},
      {"far depth unclamped", 2, 0.5f, 0.25f, 0.25f, 0.75f, true, 1.25f, 65535},
      {"near depth clamped", -2, 0.5f, 0.25f, 0.25f, 0.75f, false, 0.25f, 16384},
      {"near depth unclamped", -2, 0.5f, 0.25f, 0.25f, 0.75f, true, -0.75f, 0},
      {.name = "sample disabled", .disable_samples = true},
      {.name = "zero-component export", .fragment_export = true},
      {.name = "discard all pixels", .fragment_export = true, .export_mask = 0},
      {.name = "discard odd columns", .fragment_export = true, .export_mask = 0x5555555555555555},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    for (uint32_t bytes : {2u, 4u}) {
      for (uint32_t comparison = 0; comparison < 8; ++comparison) {
        amdgpu::Pm4QueueState state;
        state.num_instances = 1;
        state.uconfig_registers[0x242] = 17;
        state.context_registers[5] = (3 << 16) | 3;
        state.context_registers[6] = (bytes == 2 ? 1 : 3) | (3 << 4);
        state.context_registers[8] = state.context_registers[10] = 0x2000;
        state.context_registers[0x1c] = 0x700780 | 6 | (comparison << 4);
        state.context_registers[0x198] = 2;
        state.context_registers[0x2f9] = 0x2d;
        state.context_registers[0x30e] = state.context_registers[0x30f] =
            test.disable_samples ? 0 : 0xffffffffu;
        state.context_registers[0x205] = 0x43f;
        state.context_registers[0x10f] = state.context_registers[0x110] =
            state.context_registers[0x111] = state.context_registers[0x112] =
                std::bit_cast<uint32_t>(2.0f);
        state.context_registers[0x113] = std::bit_cast<uint32_t>(test.scale);
        state.context_registers[0x114] = std::bit_cast<uint32_t>(test.offset);
        state.context_registers[gfx12 ? 0x115 : 0xb4] = std::bit_cast<uint32_t>(test.minimum);
        state.context_registers[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(test.maximum);
        state.context_registers[gfx12 ? 0x19 : 3] =
            test.disable_clamp ? (gfx12 ? 1u : 1u << 16) : 0;
        state.context_registers[0x204] = (1u << 26) | (1u << 27);
        state.context_registers[0x90] = 1 | (1 << 16);
        state.context_registers[0x91] = (3 - gfx12) | ((3 - gfx12) << 16);
        if (!gfx12) {
          state.context_registers[7] = (3 << 16) | 3;
          state.context_registers[0x10] = (bytes == 2 ? 1 : 3) | (24 << 4);
          state.context_registers[0x12] = state.context_registers[0x14] = 0x2000;
          state.context_registers[0x200] = 0x700780 | 6 | (comparison << 4);
          state.context_registers[0x1c] = 0;
          state.context_registers[0x1b4] = 2;
          state.context_registers[0x206] = 0x43f;
          state.context_registers[0x205] = 0;
        }
        state.context_registers[gfx12 ? 7 : 0x11] = test.stencil_info;
        state.context_registers[gfx12 ? 0x1c : 0x200] |= test.enable_stencil;
        for (uint32_t y = 0; y < 4; ++y)
          for (uint32_t x = 0; x < 4; ++x) {
            const auto address = gfx12 ? amdgpu::gfx12_image_address(0x200000, x, y, 4, bytes, 3)
                                       : amdgpu::gfx11_image_address(0x200000, x, y, 4, bytes, 24);
            ASSERT_TRUE(address);
            const uint32_t value = bytes == 2 ? 65535 : std::bit_cast<uint32_t>(1.0f);
            memory_.write_block(*address, {reinterpret_cast<const uint8_t *>(&value), bytes});
          }
        state.context_registers[gfx12 ? 0x215 : 0x8f] = 15;
        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        for (uint32_t i = 0; i < 3; ++i)
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(test.vertex_z),
                             std::bit_cast<uint32_t>(1.0f)});
        draw->export_lane(*wave_, 0, 20, 1,
                          {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
        auto dispatch = draw->advance(*access_);
        ASSERT_EQ(bool(dispatch), !test.disable_samples);
        // A zero-component export still carries pixel validity for late depth.
        if (dispatch) {
          if (test.fragment_export) {
            initialize_fragment(draw);
            wave_->set_exec(wave_->exec() & test.export_mask);
            execute(0, 0);
          }
          EXPECT_FALSE(draw->advance(*access_));
        }
        const float expected = bytes == 2 ? test.expected_d16 / 65535.0f : test.expected;
        const bool comparisons[] = {false,        expected < 1,  expected == 1, expected <= 1,
                                    expected > 1, expected != 1, expected >= 1, true};
        const bool pass = !test.disable_samples && comparisons[comparison];
        const uint32_t expected_bits =
            bytes == 2 ? test.expected_d16 : std::bit_cast<uint32_t>(expected);
        for (uint32_t y = 0; y < 4; ++y)
          for (uint32_t x = 0; x < 4; ++x) {
            const auto address = gfx12 ? amdgpu::gfx12_image_address(0x200000, x, y, 4, bytes, 3)
                                       : amdgpu::gfx11_image_address(0x200000, x, y, 4, bytes, 24);
            uint32_t actual = 0;
            ASSERT_EQ(access_->read(*address, {reinterpret_cast<std::byte *>(&actual), bytes}),
                      amdgpu::VmAccessOutcome::Complete);
            const bool survives =
                !test.fragment_export || (test.export_mask & (uint64_t{1} << (x & 1)));
            const bool changed = pass && survives && x >= 1 && x < 3 && y >= 1 && y < 3;
            EXPECT_EQ(actual, changed      ? expected_bits
                              : bytes == 2 ? 65535
                                           : std::bit_cast<uint32_t>(1.0f))
                << bytes << "," << comparison << "," << x << "," << y;
          }
      }
    }
  }
}

TEST_P(GraphicsExportTest, DepthMipAndArrayViewsPreserveOtherSubresources) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t width = 64, levels = 7;
  constexpr uint64_t base = 0x200000;
  for (uint32_t bytes : {2u, 4u}) {
    const uint32_t swizzle = gfx12 ? 3 : 28;
    for (uint32_t level : {0u, 1u, 3u, 6u}) {
      for (const auto &[first, relative] : {std::pair{0u, 0u}, {2u, 1u}, {2049u, 1u}, {2u, 2u}}) {
        SCOPED_TRACE(testing::Message()
                     << bytes << ',' << level << ',' << first << ',' << relative);
        const uint32_t last = first + 1;
        const auto address = [&](uint32_t mip, uint32_t layer, uint32_t x, uint32_t y) {
          const auto layout =
              amdgpu::image_mip_layout(gfx12, swizzle, bytes, width, width, levels, mip);
          const uint64_t layer_base = amdgpu::image_layer_base(
              gfx12, base + layout->offset, layout->slice_size, layer, bytes, swizzle);
          return gfx12 ? amdgpu::gfx12_image_address(layer_base, x + layout->tail_x,
                                                     y + layout->tail_y, layout->pitch, bytes,
                                                     swizzle)
                       : amdgpu::gfx11_image_address(layer_base, x + layout->tail_x,
                                                     y + layout->tail_y, layout->pitch, bytes,
                                                     swizzle);
        };
        // Populate each mip in and around the view. Only the selected subresource
        // may change, including when the view uses GFX11's high slice bits.
        for (uint32_t mip = 0; mip < levels; ++mip)
          for (uint32_t layer = first; layer <= last + 1; ++layer)
            for (uint32_t y = 0; y < std::min(4u, width >> mip); ++y)
              for (uint32_t x = 0; x < std::min(4u, width >> mip); ++x) {
                const uint32_t value = bytes == 2 ? 65535 : std::bit_cast<uint32_t>(1.0f);
                ASSERT_TRUE(address(mip, layer, x, y));
                memory_.write_block(*address(mip, layer, x, y),
                                    {reinterpret_cast<const uint8_t *>(&value), bytes});
              }
        auto state = rectangle_state();
        auto &ctx = state.context_registers;
        ctx[gfx12 ? 5 : 7] = (width - 1) | ((width - 1) << 16);
        ctx[gfx12 ? 6 : 0x10] =
            (bytes == 2 ? 1 : 3) | (swizzle << 4) | ((levels - 1) << (gfx12 ? 15 : 16));
        if (gfx12) {
          ctx[1] = first | (last << 16);
          ctx[2] = level << 26;
          ctx[8] = ctx[10] = base >> 8;
        } else {
          ctx[2] = first | ((last & 0x7ff) << 13) | ((last >> 11) << 30) | (level << 26);
          ctx[0x12] = ctx[0x14] = base >> 8;
        }
        ctx[gfx12 ? 0x1c : 0x200] = 6 | (7 << 4); // ALWAYS, write enabled.
        ctx[gfx12 ? 0x206 : 0x207] = 1u << 18;    // Layer export enabled.
        const float half_extent = float(width >> level) * 0.5f;
        ctx[0x10f] = ctx[0x110] = ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(half_extent);
        ctx[0x113] = ctx[gfx12 ? 0x116 : 0xb5] = std::bit_cast<uint32_t>(1.0f);
        ctx[0x204] = (1u << 26) | (1u << 27);
        const uint32_t end = std::min(4u, width >> level) - gfx12;
        ctx[0x91] = end | (end << 16);
        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        for (uint32_t i = 0; i < 3; ++i) {
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(0.25f), std::bit_cast<uint32_t>(1.0f)});
          draw->export_lane(*wave_, i, 13, 4, {0, 0, relative, 0});
        }
        draw->export_lane(*wave_, 0, 20, 1,
                          {(1u << (gfx12 ? 9 : 10)) | (2u << (gfx12 ? 18 : 20)), 0, 0, 0});
        const auto dispatch = draw->advance(*access_);
        ASSERT_EQ(bool(dispatch), relative <= last - first);
        if (dispatch) {
          EXPECT_FALSE(draw->advance(*access_));
        }
        for (uint32_t mip = 0; mip < levels; ++mip)
          for (uint32_t layer = first; layer <= last + 1; ++layer)
            for (uint32_t y = 0; y < std::min(4u, width >> mip); ++y)
              for (uint32_t x = 0; x < std::min(4u, width >> mip); ++x) {
                uint32_t actual = 0;
                ASSERT_EQ(access_->read(*address(mip, layer, x, y),
                                        {reinterpret_cast<std::byte *>(&actual), bytes}),
                          amdgpu::VmAccessOutcome::Complete);
                const bool changed =
                    relative <= last - first && mip == level && layer == first + relative;
                const uint32_t expected = bytes == 2
                                              ? (changed ? 16384 : 65535)
                                              : std::bit_cast<uint32_t>(changed ? 0.25f : 1.0f);
                EXPECT_EQ(actual, expected) << mip << ',' << layer << ',' << x << ',' << y;
              }
      }
    }
  }
}

TEST_P(GraphicsExportTest, HardwareInterpolationEncodingUsesVgprSelectors) {
  // RADV's triangle fragment shader uses this word pair on both physical cards.
  const std::array<uint32_t, 4> words{0xcd000205, 0x040a0102, 0, 0};
  auto decoded = decoder_->decode(words.data());
  ASSERT_FALSE(decoded.failed());
  std::unique_ptr<Instruction> instruction(std::move(decoded).value());
  ASSERT_EQ(instruction->num_src_operands(), 3u);
  EXPECT_EQ(instruction->src_operand(0)->unified_vgpr_index(), 2u);
  EXPECT_EQ(instruction->src_operand(1)->unified_vgpr_index(), 0u);
  EXPECT_EQ(instruction->src_operand(2)->unified_vgpr_index(), 2u);
}

TEST_P(GraphicsExportTest, ParameterLoadOutOfRangeDestinationClearsExec) {
  wave_->set_m0(0);
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    run(rdna4::build_vdsdir(0, {.vdst = 255}));
  else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
    run(rdna3_5::build_ldsdir(0, {.vdst = 255}));
  else
    run(rdna3::build_ldsdir(0, {.vdst = 255}));
  EXPECT_EQ(wave_->exec(), 0u);
  EXPECT_FALSE(wave_->instruction_execution_failed());
}

TEST_P(GraphicsExportTest, HalfInterpolationMatchesPhysicalModesAndQuadSources) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  // Captured on GFX1100/1201. These distinguish source half selection, F32
  // versus direct-F16 rounding, RTZ overrides, signed zeros, denormals, NaN
  // priority/quieting and the older DX10_CLAMP requirement.
  struct Case {
    uint8_t opcode, opsel, neg;
    bool clamp;
    uint32_t mode;
    std::array<uint32_t, 6> expected;
  };
  const Case cases[] = {
      {2,
       0,
       0,
       false,
       0xf0,
       {0xbf803004u, 0x33800000u, gfx12 ? 0xffc00456u : 0xff800456u, 0x7fc2a000u, 0x7fc2a000u,
        gfx12 ? 0xffc00456u : 0xff800456u}},
      {2,
       1,
       1,
       false,
       0x0,
       {0xc77ffffcu, 0x00000000u, gfx12 ? 0xffc00456u : 0xff800456u, 0x7fc2a000u,
        gfx12 ? 0xffc4a000u : 0xff84a000u, gfx12 ? 0xffc00456u : 0xff800456u}},
      {2,
       4,
       4,
       false,
       0x1,
       {0xff800000u, 0xff800000u, gfx12 ? 0xffc00456u : 0xff800456u, 0x7f800000u, 0x7fc2a000u,
        gfx12 ? 0xffc00456u : 0xff800456u}},
      {2,
       5,
       7,
       false,
       0x2,
       {0xff800000u, 0xff800000u, gfx12 ? 0x7fc00456u : 0x7f800456u, 0x40002802u, 0xbeaad556u,
        gfx12 ? 0x7fc00456u : 0x7f800456u}},
      {2,
       0,
       7,
       true,
       0x3ff,
       {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {2,
       0,
       0,
       true,
       0x0,
       {0x00000000u, 0x00000000u, gfx12 ? 0x00000000u : 0xff800456u,
        gfx12 ? 0x00000000u : 0x7fc2a000u, gfx12 ? 0x00000000u : 0x7fc2a000u,
        gfx12 ? 0x00000000u : 0xff800456u}},
      {2,
       5,
       0,
       false,
       0x2f0,
       {0x7f800000u, 0x7f800000u, 0xffc00456u, 0x3a002000u, 0xbeaad554u, 0xffc00456u}},
      {2,
       4,
       0,
       false,
       0x3,
       {0x7f800000u, 0x7f800000u, gfx12 ? 0xffc00456u : 0xff800456u, 0x7f800000u, 0x7fc2a000u,
        gfx12 ? 0xffc00456u : 0xff800456u}},
      {3,
       0,
       0,
       false,
       0xf0,
       {0x12343c00u, 0x12348000u, 0x1234fe00u, 0x12347e15u, 0x1234fe25u, 0x1234fe00u}},
      {3,
       1,
       1,
       false,
       0x0,
       {0x1234fc00u, 0x1234fe00u, 0x1234fe15u, 0x12343c02u, 0x12340000u, 0x1234fe00u}},
      {3,
       8,
       4,
       false,
       0x4,
       {0xbc005678u, 0x00005678u, 0xfe005678u, 0x7e155678u, 0xfe255678u, 0xfe005678u}},
      {3,
       9,
       7,
       false,
       0x8,
       {0x7c005678u, 0xfe005678u, 0xfe155678u, 0xbc025678u, 0x80005678u, 0x7e005678u}},
      {3,
       0,
       7,
       true,
       0x3ff,
       {0x12340000u, 0x12340000u, 0x12340000u, 0x12340000u, 0x12340000u, 0x12340000u}},
      {3,
       0,
       0,
       true,
       0x0,
       {0x12343c00u, 0x12340000u, gfx12 ? 0x12340000u : 0x1234fe00u,
        gfx12 ? 0x12340000u : 0x12347e15u, gfx12 ? 0x12340000u : 0x1234fe25u,
        gfx12 ? 0x12340000u : 0x1234fe00u}},
      {3,
       9,
       0,
       false,
       0x2f0,
       {0x7c005678u, 0x7c005678u, 0x7e155678u, 0xbc025678u, 0x00005678u, 0xfe005678u}},
      {3,
       8,
       0,
       false,
       0xc,
       {0x3c005678u, 0x80005678u, 0xfe005678u, 0x7e155678u, 0xfe255678u, 0xfe005678u}},
      {4,
       0,
       0,
       false,
       0xf0,
       {0xbf803003u, 0x337fffffu, gfx12 ? 0xffc00456u : 0xff800456u, 0x7fc2a000u, 0x7fc2a000u,
        gfx12 ? 0xffc00456u : 0xff800456u}},
      {4,
       1,
       1,
       false,
       0x0,
       {0xc77ffffcu, 0x00000000u, gfx12 ? 0xffc00456u : 0xff800456u, 0x7fc2a000u,
        gfx12 ? 0xffc4a000u : 0xff84a000u, gfx12 ? 0xffc00456u : 0xff800456u}},
      {4,
       4,
       4,
       false,
       0x1,
       {0xff800000u, 0xff800000u, gfx12 ? 0xffc00456u : 0xff800456u, 0x7f800000u, 0x7fc2a000u,
        gfx12 ? 0xffc00456u : 0xff800456u}},
      {4,
       5,
       7,
       false,
       0x2,
       {0xff800000u, 0xff800000u, gfx12 ? 0x7fc00456u : 0x7f800456u, 0x40002802u, 0xbeaad555u,
        gfx12 ? 0x7fc00456u : 0x7f800456u}},
      {4,
       0,
       7,
       true,
       0x3ff,
       {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}},
      {4,
       0,
       0,
       true,
       0x0,
       {0x00000000u, 0x00000000u, gfx12 ? 0x00000000u : 0xff800456u,
        gfx12 ? 0x00000000u : 0x7fc2a000u, gfx12 ? 0x00000000u : 0x7fc2a000u,
        gfx12 ? 0x00000000u : 0xff800456u}},
      {4,
       5,
       0,
       false,
       0x2f0,
       {0x7f800000u, 0x7f800000u, 0xffc00456u, 0x3a002000u, 0xbeaad553u, 0xffc00456u}},
      {4,
       4,
       0,
       false,
       0x3,
       {0x7f800000u, 0x7f800000u, gfx12 ? 0xffc00456u : 0xff800456u, 0x7f800000u, 0x7fc2a000u,
        gfx12 ? 0xffc00456u : 0xff800456u}},
      {5,
       0,
       0,
       false,
       0xf0,
       {0x12343c00u, 0x12348000u, 0x1234fe00u, 0x12347e15u, 0x1234fe25u, 0x1234fe00u}},
      {5,
       1,
       1,
       false,
       0x0,
       {0x1234fc00u, 0x1234fe00u, 0x1234fe15u, 0x12343c01u, 0x12340000u, 0x1234fe00u}},
      {5,
       8,
       4,
       false,
       0x4,
       {0xbc005678u, 0x00005678u, 0xfe005678u, 0x7e155678u, 0xfe255678u, 0xfe005678u}},
      {5,
       9,
       7,
       false,
       0x8,
       {0x7c005678u, 0xfe005678u, 0xfe155678u, 0xbc015678u, 0x00005678u, 0x7e005678u}},
      {5,
       0,
       7,
       true,
       0x3ff,
       {0x12340000u, 0x12340000u, 0x12340000u, 0x12340000u, 0x12340000u, 0x12340000u}},
      {5,
       0,
       0,
       true,
       0x0,
       {0x12343c00u, 0x12340000u, gfx12 ? 0x12340000u : 0x1234fe00u,
        gfx12 ? 0x12340000u : 0x12347e15u, gfx12 ? 0x12340000u : 0x1234fe25u,
        gfx12 ? 0x12340000u : 0x1234fe00u}},
      {5,
       9,
       0,
       false,
       0x2f0,
       {0x7c005678u, 0x7c005678u, 0x7e155678u, 0xbc015678u, 0x00005678u, 0xfe005678u}},
      {5,
       8,
       0,
       false,
       0xc,
       {0x3c005678u, 0x80005678u, 0xfe005678u, 0x7e155678u, 0xfe255678u, 0xfe005678u}},
  };
  constexpr std::array<uint32_t, 6> lanes{0, 3, 7, 16, 20, 31};
  constexpr uint16_t halves[] = {0x3c01, 0xbc01, 0x0001, 0x8001, 0x7bff, 0x7c00, 0x7e15, 0xfc25};
  constexpr uint32_t factors[] = {0x3f801000, 0xbf801000, 0x00800000, 0x00000001,
                                  0x3eaaaaab, 0x7f800000, 0x7fc00123, 0xff800456};
  constexpr uint32_t addends[] = {0x3f800001, 0xbf800001, 0x33000000, 0xb3000000,
                                  0x80000000, 0x00000001, 0x7f800123, 0xffc00456};
  const auto packed = [&](uint32_t lane) {
    const uint32_t i = lane / 4 + lane % 4;
    return halves[i % 8] | (uint32_t(halves[(i + 3) % 8]) << 16);
  };
  uint64_t mask = 0;
  for (uint32_t lane : lanes)
    mask |= uint64_t{1} << lane;
  wave_->set_exec(mask);
  for (const auto &test : cases) {
    SCOPED_TRACE(testing::Message()
                 << unsigned(test.opcode) << ',' << unsigned(test.opsel) << ',' << test.mode);
    wave_->set_mode_raw(test.mode);
    for (uint32_t lane = 0; lane < 32; ++lane) {
      wave_->debug_write_vgpr(0, lane, packed(lane));
      wave_->debug_write_vgpr(1, lane, factors[lane % 8]);
      wave_->debug_write_vgpr(
          2, lane, test.opcode & 1 ? addends[(lane / 4 + lane % 4) % 8] : packed(lane + 8));
    }
    // Destination aliases the quad coefficient. A half write preserves the
    // other half, and inactive helper lanes remain available to later lanes.
    if (gfx12)
      run(rdna4::build_vinterp(test.opcode, {.vdst = 0,
                                             .opsel = test.opsel,
                                             .clamp = test.clamp,
                                             .src0 = 256,
                                             .src1 = 257,
                                             .src2 = 258,
                                             .neg = test.neg}));
    else
      run(rdna3::build_vinterp(test.opcode, {.vdst = 0,
                                             .op_sel = test.opsel,
                                             .clamp = test.clamp,
                                             .src0 = 256,
                                             .src1 = 257,
                                             .src2 = 258,
                                             .neg = test.neg}));
    ASSERT_FALSE(wave_->instruction_execution_failed());
    for (uint32_t i = 0; i < lanes.size(); ++i) {
      const uint32_t lane = lanes[i];
      uint32_t expected = test.expected[i];
      if (test.opcode & 1) {
        const uint32_t keep = test.opsel & 8 ? 0xffff : 0xffff0000;
        expected = (expected & ~keep) | (packed(lane) & keep);
      }
      EXPECT_EQ(wave_->debug_read_vgpr(0, lane), expected) << lane;
    }
    EXPECT_EQ(wave_->debug_read_vgpr(0, 1), packed(1));
  }
}

TEST_P(GraphicsExportTest, InterpolationPreservesHardwareNanPayloads) {
  wave_->set_exec(15);
  for (uint32_t mode = 0; mode < 8; ++mode) {
    const uint32_t ieee = mode & 1, dx10 = (mode >> 1) & 1;
    const uint8_t clamp = mode >> 2;
    wave_->set_mode_raw((ieee << 9) | (dx10 << 8));
    wave_->debug_write_vgpr(0, 0, 0xffc01234u);
    wave_->debug_write_vgpr(0, 1, 0x7f801abcu);
    for (uint32_t lane = 0; lane < 4; ++lane)
      wave_->debug_write_vgpr(1, lane, 0x3f800000u);
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
      run(rdna4::build_vinterp(0,
                               {.vdst = 0, .clamp = clamp, .src0 = 256, .src1 = 257, .src2 = 256}));
    else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      run(rdna3_5::build_vinterp(
          0, {.vdst = 0, .clamp = clamp, .src0 = 256, .src1 = 257, .src2 = 256}));
    else
      run(rdna3::build_vinterp(0,
                               {.vdst = 0, .clamp = clamp, .src0 = 256, .src1 = 257, .src2 = 256}));
    const uint32_t expected = (clamp && (dx10 || GetParam() == ROCJITSU_CODE_ARCH_RDNA4)) ? 0u
                              : (ieee || GetParam() == ROCJITSU_CODE_ARCH_RDNA4) ? 0x7fc01abcu
                                                                                 : 0x7f801abcu;
    for (uint32_t lane = 0; lane < 4; ++lane)
      EXPECT_EQ(wave_->debug_read_vgpr(0, lane), expected);
  }
}

TEST(GraphicsRasterMathTest, ParameterDifferencesPreserveReferenceNan) {
  const auto difference = [](uint32_t a, uint32_t b) {
    return std::bit_cast<uint32_t>(
        amdgpu::raster::attribute_difference(std::bit_cast<float>(a), std::bit_cast<float>(b)));
  };
  EXPECT_EQ(difference(0x7fc01234u, 0xff801abcu), 0xff801abcu);
  EXPECT_EQ(difference(0x3f800000u, 0x7f801abcu), 0x7f801abcu);
  EXPECT_EQ(difference(0xff801abcu, 0x3f800000u), 0xff801abcu);
}

TEST_P(GraphicsExportTest, SecondInterpolationStepUsesP20AndModifiers) {
  wave_->set_exec(15);
  wave_->set_mode_raw(0xf0);
  for (uint32_t lane = 0; lane < 4; ++lane) {
    wave_->debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(4.0f));
    wave_->debug_write_vgpr(1, lane, std::bit_cast<uint32_t>(0.25f));
    wave_->debug_write_vgpr(2, lane, std::bit_cast<uint32_t>(float(lane)));
  }
  // Clamp(-P20 * J + tmp) = clamp(lane - 1); v0 aliases the quad coefficient.
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    run(rdna4::build_vinterp(
        1, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258, .neg = 1}));
  else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
    run(rdna3_5::build_vinterp(
        1, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258, .neg = 1}));
  else
    run(rdna3::build_vinterp(
        1, {.vdst = 0, .clamp = 1, .src0 = 256, .src1 = 257, .src2 = 258, .neg = 1}));
  EXPECT_FALSE(wave_->instruction_execution_failed());
  for (uint32_t lane = 0; lane < 4; ++lane)
    EXPECT_EQ(wave_->debug_read_vgpr(0, lane), lane < 2 ? 0u : 0x3f800000u);
}

TEST_P(GraphicsExportTest, InterpolationSnapshotsQuadSourcesWhenDestinationAliases) {
  wave_->set_exec(0xfd); // Lane one is inactive but supplies P10 to its quad.
  wave_->set_mode_raw(0xf0);
  for (uint32_t lane = 0; lane < 8; ++lane) {
    float coefficient = lane % 4 == 0 ? 1.0f : lane % 4 == 1 ? 2.0f : 4.0f;
    if (lane >= 4)
      coefficient *= 2.0f;
    wave_->debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(coefficient));
    wave_->debug_write_vgpr(1, lane, std::bit_cast<uint32_t>(0.25f));
  }
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    run(rdna4::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
  else if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
    run(rdna3_5::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
  else
    run(rdna3::build_vinterp(0, {.vdst = 0, .src0 = 256, .src1 = 257, .src2 = 256}));
  EXPECT_FALSE(wave_->instruction_execution_failed());
  for (uint32_t lane = 0; lane < 8; ++lane) {
    const float expected = lane == 1 ? 2.0f : lane < 4 ? 1.5f : 3.0f;
    EXPECT_EQ(wave_->debug_read_vgpr(0, lane), std::bit_cast<uint32_t>(expected)) << lane;
  }
}

TEST_P(GraphicsExportTest, InterpolationPreservesHostEnvironmentForRegisterObservers) {
  class Observer final : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("interpolation_environment") {}
    using State = std::array<uint32_t, 3>;
    static State capture() {
      State state{uint32_t(std::fegetround()), uint32_t(std::fetestexcept(FE_ALL_EXCEPT)), 0};
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
      state[2] = _mm_getcsr();
#endif
      return state;
    }
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *, uint32_t, uint64_t, uint8_t) override {
      states.push_back(capture());
      ++reads;
    }
    void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *, uint32_t, uint64_t, uint8_t) override {
      states.push_back(capture());
      ++writes;
    }
    std::vector<State> states;
    uint32_t reads = 0, writes = 0;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Observer>();
  auto *recorded = observer.get();
  group->add(std::move(observer));
  cu_->set_plugin_group(group);
  struct Case {
    uint32_t a, b, c, mode, expected;
    bool clamp = false;
  };
  const Case cases[] = {
      // Midpoint rounding follows guest MODE even with hostile host rounding.
      {0x3f800000, 0x33800000, 0x3f800000, 0x30, 0x3f800000},
      {0x3f800000, 0x33800000, 0x3f800000, 0x31, 0x3f800001},
      {0xbf800000, 0x33800000, 0xbf800000, 0x32, 0xbf800001},
      {0xbf800000, 0x33800000, 0xbf800000, 0x33, 0xbf800000},
      // Input and output flushing are independent of host DAZ/FTZ.
      {0x3f800000, 0x00000001, 0x00000000, 0x30, 0x00000001},
      {0x3f800000, 0x00000001, 0x00000000, 0x20, 0x00000000},
      {0x00800000, 0x3f000000, 0x00000000, 0x10, 0x00000000},
      // Tiny rounding to minimum normal exercises the nested precision guard.
      {0x00800000, 0x3f7fffff, 0x00000000, 0x00, 0x00000000},
      {0x00800000, 0x3f7fffff, 0x00000000, 0x30, 0x00800000},
      {0x00800000, 0x3f7fffff, 0x00000000, 0x32, 0x007fffff},
      {0x7f801234, 0x3f800000, 0x00000000, 0x30, 0x7f801234},
      {0x00000000, 0x7f800000, 0x00000000, 0x30, 0xffc00000},
      {0x7f801234, 0x3f800000, 0x00000000, 0x130, 0x00000000, true},
      {0x3f800000, 0x33800000, 0x3f800000, 0x31, 0x3f800000, true},
  };
  for (bool second : {false, true})
    for (bool ieee : {false, true})
      for (const auto &test : cases) {
        SCOPED_TRACE(testing::Message() << second << ',' << ieee << ',' << test.mode << ','
                                        << test.a << ',' << test.b << ',' << test.c);
        wave_->set_exec(0xfd); // Inactive lane one still supplies the quad coefficient.
        wave_->set_mode_raw(test.mode | (uint32_t(ieee) << 9));
        for (uint32_t lane = 0; lane < 8; ++lane) {
          wave_->debug_write_vgpr(0, lane, test.a);
          wave_->debug_write_vgpr(1, lane, test.b);
          wave_->debug_write_vgpr(2, lane, test.c);
        }
        recorded->states.clear();
        recorded->reads = recorded->writes = 0;
        Observer::State before, after;
        {
          amdgpu::fp_mode::ScopedEnvironment restore_host(0);
          std::fesetround(FE_UPWARD);
          std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
          // Preserve sticky flags and an enabled invalid-operation trap as well.
          _mm_setcsr((_mm_getcsr() | (1u << 6) | (1u << 15)) & ~(1u << 7));
#endif
          before = Observer::capture();
          amdgpu::execute_graphics_interp(*wave_, 0, {0, 1, 2}, second, 0, test.clamp, 0);
          after = Observer::capture();
        }
        EXPECT_EQ(after, before);
        ASSERT_EQ(recorded->reads, 21u);
        ASSERT_EQ(recorded->writes, 7u);
        for (const auto &state : recorded->states)
          EXPECT_EQ(state, before);
        uint32_t expected = test.expected;
        if (expected == 0x7f801234 && (ieee || GetParam() == ROCJITSU_CODE_ARCH_RDNA4))
          expected |= 0x00400000;
        for (uint32_t lane = 0; lane < 8; ++lane)
          EXPECT_EQ(wave_->debug_read_vgpr(0, lane), lane == 1 ? test.a : expected) << lane;
      }
}

TEST_P(GraphicsExportTest, PreservesBitsAndSelectsActiveLanesAndComponents) {
  wave_->set_exec(uint64_t{1} | (uint64_t{1} << 31));
  wave_->debug_write_vgpr(3, 0, 0x7fc12345);
  wave_->debug_write_vgpr(9, 0, 0x80000000);
  wave_->debug_write_vgpr(3, 31, 0x3c003800);
  wave_->debug_write_vgpr(9, 31, 0xfedcba98);
  execute(12, 5);
  EXPECT_FALSE(wave_->instruction_execution_failed());
  ASSERT_EQ(collector_->exports.size(), 2u);
  EXPECT_EQ(collector_->exports[0].lane, 0u);
  EXPECT_EQ(collector_->exports[1].lane, 31u);
  EXPECT_EQ(collector_->exports[1].target, 12u);
  EXPECT_EQ(collector_->exports[1].mask, 5u);
  EXPECT_EQ(collector_->exports[0].values, (std::array<uint32_t, 4>{0x7fc12345, 0, 0x80000000, 0}));
  EXPECT_EQ(collector_->exports[1].values, (std::array<uint32_t, 4>{0x3c003800, 0, 0xfedcba98, 0}));
}

TEST_P(GraphicsExportTest, RejectsEnabledUnallocatedSourceBeforeExporting) {
  execute(0, 15);
  EXPECT_TRUE(wave_->instruction_execution_failed());
  EXPECT_TRUE(collector_->exports.empty());
}

TEST_P(GraphicsExportTest, UnsupportedRowModeFailsSubmission) {
  execute(12, 1, true);
  EXPECT_TRUE(wave_->instruction_execution_failed());
  EXPECT_TRUE(collector_->exports.empty());
}

TEST_P(GraphicsExportTest, MissingExportDestinationFailsExecution) {
  wave_->set_graphics_stage(nullptr);
  execute(0, 1);
  EXPECT_TRUE(wave_->instruction_execution_failed());
}

TEST_P(GraphicsExportTest, SkipExportAndEmptyExecDoNotAccessSources) {
  wave_->set_graphics_stage(nullptr);
  wave_->set_status_raw(wave_->status_raw() | (1u << 18));
  execute(12, 15, true);
  EXPECT_FALSE(wave_->instruction_execution_failed());
  wave_->set_status_raw(wave_->status_raw() & ~(1u << 18));
  wave_->set_exec(0);
  execute(12, 15, true);
  EXPECT_FALSE(wave_->instruction_execution_failed());
}

TEST_P(GraphicsExportTest, ExpandedHtileProbeRequiresCallerOptInAndQualifiedBacking) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  for (bool clear : {false, true}) {
    for (uint32_t variant = 0; variant < 3; ++variant) {
      SCOPED_TRACE(testing::Message() << variant << ", clear=" << clear);
      constexpr uint32_t width = 16, height = 8;
      constexpr uint64_t metadata = 0x100000, depth = 0x200000;
      auto backing = std::make_shared<GraphicsBatchMemory>();
      amdgpu::GpuVm vm;
      const auto handle = vm.register_address_space(7, backing, backing);
      const auto access = vm.snapshot(handle);
      ASSERT_TRUE(access);
      const uint32_t bytes = variant == 2 ? 2 : 4;
      for (uint32_t x = 0; x < width; x += 8) {
        const auto address =
            *amdgpu::gfx11_metadata_address(metadata, x, 0, width, height, bytes, 24, true);
        const uint32_t key = clear ? 0 : 0xfffc000f;
        std::memcpy(backing->bytes.data() + address, &key, sizeof(key));
      }
      auto state = rectangle_state();
      auto &ctx = state.context_registers;
      ctx[0x200] = 2 | (7 << 4);
      ctx[7] = (width - 1) | ((height - 1) << 16);
      ctx[0x10] = (bytes == 4 ? 3 : 1) | (24 << 4) | (1u << 29);
      ctx[0x12] = ctx[0x14] = depth >> 8;
      ctx[0x2af] = 1u << 18;
      ctx[5] = metadata >> 8;
      ctx[0x10f] = ctx[0x110] = std::bit_cast<uint32_t>(width / 2.0f);
      ctx[0x111] = ctx[0x112] = std::bit_cast<uint32_t>(height / 2.0f);
      ctx[0xb5] = std::bit_cast<uint32_t>(1.0f);
      ctx[0x91] = width | (height << 16);
      amdgpu::GraphicsDraw draw(state, GetParam(), 3);
      export_rectangle_vertices(draw);
      // The default is the caller's observer/debug path. Even when opted in,
      // this custom translator refuses the lease and retains its ordinary
      // reads.
      if (variant == 0) {
        EXPECT_TRUE(draw.advance(*access));
      } else {
        EXPECT_TRUE(draw.advance(*access, nullptr, 1, true));
      }
      EXPECT_EQ(backing->ram_lease_requests, variant == 1 ? 1u : 0u);
      EXPECT_EQ(backing->reads, 2u);
      EXPECT_EQ(backing->writes, clear ? width * height + 2 : 0u);
    }
  }
}

TEST_P(GraphicsExportTest, MetadataAttachmentsPreserveDrawsAndRejectedState) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  struct Case {
    const char *name;
    bool valid_viewport = true;
    float min_depth = 0;
    float max_depth = 1;
    bool scratch = false;
    bool reject = false;
  };
  const Case cases[] = {
      {.name = "valid state"},
      {.name = "invalid viewport", .valid_viewport = false, .reject = true},
      {.name = "nonfinite depth range",
       .max_depth = std::bit_cast<float>(0x7fc00000u),
       .reject = true},
      {.name = "inverted depth range", .min_depth = 1, .max_depth = 0, .reject = true},
      {.name = "fragment scratch", .scratch = true},
  };
  for (uint32_t target : {0u, 7u})
    for (bool aligned : {true, false}) {
      for (const auto &test : cases) {
        SCOPED_TRACE(testing::Message() << "target=" << target << ", pipe_aligned=" << aligned);
        SCOPED_TRACE(test.name);
        constexpr uint32_t width = 17, height = 13;
        constexpr uint64_t color_base = 0x200000, dcc = 0x100000;
        constexpr uint64_t depth_base = 0x400000, htile = 0x300000;
        for (uint32_t i = 0; i < 16384; ++i)
          memory_.write8(dcc + i, 0xff);
        for (uint32_t y = 0; y < height; y += 8) {
          for (uint32_t x = 0; x < width; x += 8) {
            memory_.write8(
                *amdgpu::gfx11_metadata_address(dcc, x, y, width, height, 4, 27, false, aligned),
                8);
            memory_.write32(
                *amdgpu::gfx11_metadata_address(htile, x, y, width, height, 4, 24, true),
                0x55555550);
          }
        }
        for (uint32_t y = 0; y < height; ++y) {
          for (uint32_t x = 0; x < width; ++x) {
            memory_.write32(*amdgpu::gfx11_image_address(color_base, x, y, width, 4, 27),
                            0x12345678);
            memory_.write32(*amdgpu::gfx11_image_address(depth_base, x, y, width, 4, 24),
                            0x12345678);
          }
        }
        amdgpu::Pm4QueueState state;
        state.num_instances = 1;
        state.uconfig_registers[0x242] = 17;
        auto &context = state.context_registers;

        const uint32_t block = 0x318 + 15 * target;
        context[block + 4] = 10;
        context[0x3b0 + target] = (height - 1) | ((width - 1) << 14);
        context[0x3b8 + target] = (27 << 14) | (uint32_t(aligned) << 30);
        context[block + 6] = 1u << 22;
        context[block + 13] = dcc >> 8;
        context[block] = color_base >> 8;
        context[0x8e] = 15u << (4 * target);
        context[0x8f] = 15u << (4 * target);
        context[0x1c5] = 4;
        context[0x1b4] = 2;
        context[0x2f9] = 0x2d;
        context[0x206] = test.valid_viewport ? 0x43f : 0;
        context[0x202] = 0xcc0010;
        context[0x10f] = context[0x110] = std::bit_cast<uint32_t>(width / 2.0f);
        context[0x111] = context[0x112] = std::bit_cast<uint32_t>(height / 2.0f);
        context[0x113] = std::bit_cast<uint32_t>(1.0f);
        context[0xb4] = std::bit_cast<uint32_t>(test.min_depth);
        context[0xb5] = std::bit_cast<uint32_t>(test.max_depth);
        state.sh_registers[0xb] = test.scratch;
        context[0x90] = 16 | (8 << 16);
        context[0x91] = 17 | (9 << 16);
        context[0x30e] = context[0x30f] = 0xffffffff;
        context[0x200] = 6 | (1 << 4); // Depth test LESS and write enabled.
        context[7] = (width - 1) | ((height - 1) << 16);
        context[0x10] = 3 | (24 << 4) | (1u << 29); // D32 with HTILE.
        context[0x12] = context[0x14] = depth_base >> 8;
        context[0x2af] = 1u << 18;
        context[5] = htile >> 8;
        context[0xb] = std::bit_cast<uint32_t>(0.375f);

        auto draw = std::make_shared<amdgpu::GraphicsDraw>(state, GetParam(), 3);
        for (uint32_t i = 0; i < 3; ++i)
          draw->export_lane(*wave_, i, 12, 15,
                            {std::bit_cast<uint32_t>(i == 2 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(i == 1 ? 1.0f : -1.0f),
                             std::bit_cast<uint32_t>(0.25f), std::bit_cast<uint32_t>(1.0f)});
        draw->export_lane(*wave_, 0, 20, 1, {(1u << 10) | (2u << 20), 0, 0, 0});
        const auto color_address = *amdgpu::gfx11_image_address(color_base, 16, 8, width, 4, 27);
        const auto depth_address = *amdgpu::gfx11_image_address(depth_base, 16, 8, width, 4, 24);
        const auto color_key =
            *amdgpu::gfx11_metadata_address(dcc, 16, 8, width, height, 4, 27, false, aligned);
        const auto depth_key =
            *amdgpu::gfx11_metadata_address(htile, 16, 8, width, height, 4, 24, true);
        if (test.reject) {
          EXPECT_THROW(draw->advance(*access_), std::runtime_error);
          EXPECT_EQ(memory_.read32(color_address), 0x12345678u);
          EXPECT_EQ(memory_.read32(depth_address), 0x12345678u);
          EXPECT_EQ(memory_.read8(color_key), 8);
          EXPECT_EQ(memory_.read32(depth_key), 0x55555550u);
          continue;
        }
        ASSERT_TRUE(draw->advance(*access_));
        EXPECT_EQ(memory_.read32(color_address), 0xff000000u);
        EXPECT_EQ(memory_.read32(depth_address), std::bit_cast<uint32_t>(0.375f));
        EXPECT_EQ(memory_.read8(color_key), 0xff);
        EXPECT_EQ(memory_.read32(depth_key), 0xfffc000fu);
        initialize_fragment(draw);
        for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
          draw->export_lane(*wave_, lane, 0, 3, {0x00003c00, 0x3c000000, 0, 0});
        EXPECT_FALSE(draw->advance(*access_));
        EXPECT_EQ(memory_.read32(color_address), 0xff0000ffu);
        EXPECT_EQ(memory_.read32(depth_address), std::bit_cast<uint32_t>(0.25f));
        amdgpu::materialize_gfx11_dcc(*access_, color_base, dcc, 16, 8, width, height, 4, 27,
                                      aligned);
        amdgpu::materialize_gfx11_htile(*access_, depth_base, htile, 16, 8, width, height, 4, 24);
        EXPECT_EQ(memory_.read32(color_address), 0xff0000ffu);
        EXPECT_EQ(memory_.read32(depth_address), std::bit_cast<uint32_t>(0.25f));
      }
    }
}

INSTANTIATE_TEST_SUITE_P(Rdna, GraphicsExportTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

} // namespace

TEST(GraphicsImageMetadataTest, AddressesMatchAddrLibAcrossMetadataBlocks) {
  struct Case {
    uint32_t swizzle, bytes, width, x, y, pipe_xor, expected;
    bool depth, pipe_aligned;
  };
  const Case cases[] = {
      {27, 1, 420, 419, 319, 7, 1102, false, false},
      {27, 1, 4093, 3001, 2049, 31, 43589, false, false},
      {27, 1, 420, 419, 319, 7, 5133, false, true},
      {27, 1, 4093, 3001, 2049, 31, 49173, false, true},
      {27, 2, 420, 419, 319, 7, 413, false, false},
      {27, 2, 4093, 3001, 2049, 31, 75146, false, false},
      {27, 2, 420, 419, 319, 7, 5147, false, true},
      {27, 2, 4093, 3001, 2049, 31, 81962, false, true},
      {27, 4, 420, 419, 319, 7, 2618, false, false},
      {27, 4, 4093, 3001, 2049, 31, 154133, false, false},
      {27, 4, 420, 419, 319, 7, 5174, false, true},
      {27, 4, 4093, 3001, 2049, 31, 163925, false, true},
      {27, 8, 420, 419, 319, 7, 7541, false, false},
      {27, 8, 4093, 3001, 2049, 31, 283946, false, false},
      {27, 8, 420, 419, 319, 7, 1142, false, true},
      {27, 8, 4093, 3001, 2049, 31, 295061, false, true},
      {27, 16, 420, 419, 319, 7, 13290, false, false},
      {27, 16, 4093, 3001, 2049, 31, 572244, false, false},
      {27, 16, 420, 419, 319, 7, 1206, false, true},
      {27, 16, 4093, 3001, 2049, 31, 606229, false, true},
      {31, 1, 420, 419, 319, 7, 1102, false, false},
      {31, 1, 4093, 3001, 2049, 31, 43589, false, false},
      {31, 1, 420, 419, 319, 7, 5133, false, true},
      {31, 1, 4093, 3001, 2049, 31, 49173, false, true},
      {31, 2, 420, 419, 319, 7, 413, false, false},
      {31, 2, 4093, 3001, 2049, 31, 75146, false, false},
      {31, 2, 420, 419, 319, 7, 5147, false, true},
      {31, 2, 4093, 3001, 2049, 31, 81962, false, true},
      {31, 4, 420, 419, 319, 7, 2618, false, false},
      {31, 4, 4093, 3001, 2049, 31, 154133, false, false},
      {31, 4, 420, 419, 319, 7, 5174, false, true},
      {31, 4, 4093, 3001, 2049, 31, 163925, false, true},
      {31, 8, 420, 419, 319, 7, 7541, false, false},
      {31, 8, 4093, 3001, 2049, 31, 283946, false, false},
      {31, 8, 420, 419, 319, 7, 5229, false, true},
      {31, 8, 4093, 3001, 2049, 31, 295082, false, true},
      {31, 16, 420, 419, 319, 7, 13290, false, false},
      {31, 16, 4093, 3001, 2049, 31, 572244, false, false},
      {31, 16, 420, 419, 319, 7, 5338, false, true},
      {31, 16, 4093, 3001, 2049, 31, 606292, false, true},
      {24, 4, 420, 419, 319, 7, 5336, true, true},
      {24, 4, 4093, 3001, 2049, 31, 663636, true, true},
      {28, 4, 420, 419, 319, 7, 5336, true, true},
      {28, 4, 4093, 3001, 2049, 31, 663636, true, true},
  };
  for (const auto &test : cases) {
    // These single-layer offsets only require height to include the sampled y.
    // Descriptor writers mask the pipe XOR to the metadata allocation alignment.
    const uint32_t alignment = test.depth ? 131072 : test.pipe_aligned ? 16384 : 4096;
    const auto actual = amdgpu::gfx11_metadata_address(
        0x100000 | ((test.pipe_xor << 8) & (alignment - 1)), test.x, test.y, test.width, test.y + 1,
        test.bytes, test.swizzle, test.depth, test.pipe_aligned);
    EXPECT_EQ(actual, 0x100000 + test.expected)
        << test.swizzle << "," << test.bytes << "," << test.x << "," << test.y;
  }
}

TEST(GraphicsImageMetadataTest, DccMipLayoutMatchesAddrLib) {
  struct Case {
    uint32_t swizzle, bytes, width, height, levels, level;
    bool pipe_aligned;
    uint64_t offset, slice_size;
    uint32_t first_tail;
  };
  // Addr2ComputeDccInfo with GB_ADDR_CONFIG=0x545. Include odd storage
  // extents, multi-block levels, both swizzles and both pipe policies.
  constexpr Case cases[] = {
      {27, 4, 512, 512, 5, 0, true, 49152, 65536, 3},
      {27, 4, 512, 512, 5, 3, true, 0, 65536, 3},
      {27, 4, 512, 512, 5, 4, true, 0, 65536, 3},
      {27, 4, 129, 71, 8, 1, true, 16384, 49152, 2},
      {31, 16, 4093, 3001, 12, 0, true, 327680, 1114112, 6},
      {27, 8, 4093, 3001, 12, 0, false, 143360, 536576, 6},
  };
  for (const auto &c : cases) {
    const auto mip = amdgpu::gfx11_dcc_mip_layout(c.swizzle, c.bytes, c.width, c.height, c.levels,
                                                  c.level, c.pipe_aligned);
    ASSERT_TRUE(mip);
    EXPECT_EQ(mip->offset, c.offset);
    EXPECT_EQ(mip->slice_size, c.slice_size);
    EXPECT_EQ(mip->pixels.first_tail, c.first_tail);
    EXPECT_EQ(mip->enabled, c.level <= c.first_tail);
  }
  EXPECT_FALSE(amdgpu::gfx11_dcc_mip_layout(24, 4, 512, 512, 5, 0, true));
  EXPECT_FALSE(amdgpu::gfx11_dcc_mip_layout(27, 3, 512, 512, 5, 0, true));
}

TEST(GraphicsImageMetadataTest, DccClearsPreserveNeighborBlocksAndLaterWrites) {
  amdgpu::GpuMemory memory{"dcc_clear_memory"};
  amdgpu::GpuVm vm;
  const auto address_space =
      vm.register_address_space(0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory));
  const auto access = vm.snapshot(address_space);
  ASSERT_TRUE(access);
  constexpr uint64_t base = 0x200000, metadata = 0x100700;
  for (const auto &[key, expected] : {std::pair{0u, 0u},
                                      {1u, 0x76543210u},
                                      {2u, 0xffffffffu},
                                      {4u, 0x3c003c00u},
                                      {6u, 0x3f800000u},
                                      {8u, 0xff000000u},
                                      {10u, 0x00ffffffu}}) {
    const auto tag = amdgpu::gfx11_metadata_address(metadata, 8, 8, 17, 17, 4, 27, false);
    const auto neighbor = amdgpu::gfx11_metadata_address(metadata, 16, 8, 17, 17, 4, 27, false);
    memory.write8(*tag, key);
    memory.write8(*neighbor, 2);
    memory.write32(*amdgpu::gfx11_image_address(base, 8, 8, 17, 4, 27), 0x76543210);
    amdgpu::materialize_gfx11_dcc(*access, base, metadata, 11, 10, 17, 13, 4, 27);
    for (uint32_t y = 8; y < 13; ++y)
      for (uint32_t x = 8; x < 16; ++x)
        EXPECT_EQ(memory.read32(*amdgpu::gfx11_image_address(base, x, y, 17, 4, 27)), expected);
    EXPECT_EQ(memory.read8(*tag), 0xff);
    EXPECT_EQ(memory.read8(*neighbor), 2);
    const auto changed = *amdgpu::gfx11_image_address(base, 11, 10, 17, 4, 27);
    memory.write32(changed, 0x10203040);
    amdgpu::materialize_gfx11_dcc(*access, base, metadata, 11, 10, 17, 13, 4, 27);
    EXPECT_EQ(memory.read32(changed), 0x10203040u);
    memory.write8(*tag, 3);
    EXPECT_THROW(amdgpu::materialize_gfx11_dcc(*access, base, metadata, 11, 10, 17, 13, 4, 27),
                 std::runtime_error);
  }
}

TEST(GraphicsImageMetadataTest, LayerDccClearPreservesOtherLayers) {
  amdgpu::GpuMemory memory{"layered_dcc_memory"};
  amdgpu::GpuVm vm;
  const auto address_space = vm.register_address_space(
      0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
      std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory), {}, true);
  const auto access = vm.snapshot(address_space);
  ASSERT_TRUE(access);
  // AddrLib: 129x71 RGBA8 R_X, layer 3 has data at 3*131072 ^ 0x600
  // and DCC at 3*16384 ^ 0x600. Layer 2 uses XOR 0x200.
  memory.write32(0x10c600, 2);
  memory.write32(0x108200, 2);
  memory.write32(0x460600, 0);
  memory.write32(0x440200, 0x12345678);
  amdgpu::materialize_gfx11_dcc(*access, 0x400000, 0x100000, 0, 0, 129, 71, 4, 27, true, 3, 131072);
  EXPECT_EQ(memory.read32(0x460600), 0xffffffffu);
  EXPECT_EQ(memory.read32(0x440200), 0x12345678u);
  EXPECT_EQ(memory.read32(0x10c600), 0xffu);
  EXPECT_EQ(memory.read32(0x108200), 2u);
}

namespace {

// Exercise the real compatibility translator and sealed-RAM admission. Raw PTE
// edits below deliberately cover states that KfdProcess normally normalizes.
struct ExpandedHtileFixture {
  static constexpr uint64_t metadata_base = 0x100000, depth_base = 0x200000;
  static constexpr auto sealed = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
  struct Reporter final : amdgpu::MemoryFaultReporter {
    std::vector<uint64_t> addresses;
    void report_memory_fault(uint32_t, uint64_t address, amdgpu::MemoryFaultCause) override {
      addresses.push_back(address);
    }
  } reporter;
  amdgpu::GpuMemory memory{"expanded_htile"};
  KfdProcess process{7};
  std::vector<uint8_t> metadata = std::vector<uint8_t>(131072, 0x5a);
  std::vector<uint8_t> depth = std::vector<uint8_t>(262144, 0x5a);
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter{vm, &memory};
  std::optional<amdgpu::GpuVmAccess> access;
  uint32_t width, height, swizzle;

  ExpandedHtileFixture(uint32_t width = 16, uint32_t height = 8, uint32_t swizzle = 24)
      : width(width), height(height), swizzle(swizzle) {
    process.map_pages(metadata_base, metadata.data(), metadata.size(), amdgpu::Mtype::RW, sealed);
    process.map_pages(depth_base, depth.data(), depth.size(), amdgpu::Mtype::RW, sealed);
    const auto handle = adapter.register_address_space(
        7, {.page_table = &process.page_table_,
            .page_table_mutex = &process.page_table_mutex_,
            .page_table_generation = process.page_table_generation(),
            .request_mutex = process.page_table_request_mutex(),
            .mutation_epoch = process.page_table_mutation_epoch(),
            .page_table_cache_state = process.page_table_cache_state(),
            .fault_reporter = &reporter});
    access = vm.snapshot(handle);
    for (uint32_t y = 0; y < height; y += 8)
      for (uint32_t x = 0; x < width; x += 8)
        set_key(x, y, 0xfffc000f);
  }
  uint64_t key_address(uint32_t x, uint32_t y) const {
    return *amdgpu::gfx11_metadata_address(metadata_base, x, y, width, height, 4, swizzle, true);
  }
  void set_key(uint32_t x, uint32_t y, uint32_t value) {
    std::memcpy(metadata.data() + key_address(x, y) - metadata_base, &value, sizeof(value));
  }
  uint32_t key(uint32_t x, uint32_t y) const {
    uint32_t value;
    std::memcpy(&value, metadata.data() + key_address(x, y) - metadata_base, sizeof(value));
    return value;
  }
  bool probe() const {
    return amdgpu::try_gfx11_expanded_htile(*access, metadata_base, width, height, swizzle);
  }
  bool materialize_leased(uint32_t clear = 0x3f000000, bool has_stencil = false) const {
    return amdgpu::try_materialize_gfx11_htile_layer(*access, depth_base, metadata_base, width,
                                                     height, swizzle, clear, has_stencil);
  }
  void materialize(bool probe_first, uint32_t clear = 0x3f000000, bool use_lease = false,
                   bool has_stencil = false) const {
    if (probe_first && probe())
      return;
    if (use_lease && materialize_leased(clear, has_stencil))
      return;
    for (uint32_t y = 0; y < height; y += 8)
      for (uint32_t x = 0; x < width; x += 8)
        amdgpu::materialize_gfx11_htile(*access, depth_base, metadata_base, x, y, width, height, 4,
                                        swizzle, clear, has_stencil);
  }
};

} // namespace

TEST(GraphicsImageMetadataTest, ExpandedHtileProbeUsesFreshLegacyRamAndPreservesHostState) {
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (uint32_t swizzle : {24u, 28u}) {
    ExpandedHtileFixture fixture(320, 240, swizzle);
    ASSERT_TRUE(fixture.access);
    const auto metadata = fixture.metadata, depth = fixture.depth;
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      ASSERT_EQ(std::fesetround(mode), 0);
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
      errno = EDOM;
      const bool expanded = fixture.probe();
      const int observed_errno = errno, flags = std::fetestexcept(FE_ALL_EXCEPT);
      EXPECT_TRUE(expanded);
      EXPECT_EQ(observed_errno, EDOM);
      EXPECT_EQ(flags, FE_INVALID | FE_DIVBYZERO);
      EXPECT_EQ(std::fegetround(), mode);
    }
    EXPECT_EQ(fixture.metadata, metadata);
    EXPECT_EQ(fixture.depth, depth);
    EXPECT_TRUE(fixture.reporter.addresses.empty());
    const auto generation = *fixture.process.page_table_generation();
    fixture.set_key(8, 8, 0);
    EXPECT_FALSE(fixture.probe());
    EXPECT_EQ(*fixture.process.page_table_generation(), generation);
    fixture.materialize(true);
    EXPECT_EQ(fixture.key(8, 8), 0xfffc000fu);
    EXPECT_TRUE(fixture.probe());
    // Expanded metadata never needs a valid depth-pixel mapping.
    fixture.process.unmap_pages(fixture.depth_base, fixture.depth.size());
    EXPECT_TRUE(fixture.probe());
    EXPECT_NO_THROW(fixture.materialize(false));
  }
}

TEST(GraphicsImageMetadataTest, ExpandedHtileProbePreservesEveryNonExpandedKeyFallback) {
  for (uint32_t low = 0; low < 15; ++low) {
    for (uint32_t x : {0u, 8u, 16u}) {
      SCOPED_TRACE(testing::Message() << low << ',' << x);
      ExpandedHtileFixture original(24), probed(24);
      ASSERT_TRUE(original.access && probed.access);
      for (auto *fixture : {&original, &probed}) {
        fixture->set_key(0, 0, 0); // A completed clear precedes later failures.
        fixture->set_key(x, 0, 0x55555550 | low);
      }
      const auto before = probed.metadata;
      EXPECT_FALSE(probed.probe());
      EXPECT_EQ(probed.metadata, before);
      EXPECT_TRUE(probed.reporter.addresses.empty());
      if (low) {
        EXPECT_THROW(original.materialize(false), std::runtime_error);
        EXPECT_THROW(probed.materialize(true), std::runtime_error);
      } else {
        EXPECT_NO_THROW(original.materialize(false));
        EXPECT_NO_THROW(probed.materialize(true));
      }
      EXPECT_EQ(probed.metadata, original.metadata);
      EXPECT_EQ(probed.depth, original.depth);
      EXPECT_EQ(probed.reporter.addresses, original.reporter.addresses);
    }
  }
}

TEST(GraphicsImageMetadataTest, ExpandedHtileProbePreservesFaultAndAliasedStorePrefixes) {
  for (uint32_t variant = 0; variant < 3; ++variant) {
    SCOPED_TRACE(variant);
    ExpandedHtileFixture original, probed;
    ASSERT_TRUE(original.access && probed.access);
    for (auto *fixture : {&original, &probed}) {
      fixture->set_key(0, 0, 0);
      if (variant == 0) {
        // A later metadata word faults only after the first clear is published.
        fixture->process.unmap_pages(fixture->metadata_base, fixture->metadata.size());
        fixture->process.map_pages(fixture->metadata_base, fixture->metadata.data(), 4,
                                   amdgpu::Mtype::RW, fixture->sealed);
      } else if (variant == 1) {
        // One depth pixel succeeds; the next DWORD crosses a strict subextent.
        fixture->process.unmap_pages(fixture->depth_base, fixture->depth.size());
        fixture->process.map_pages(fixture->depth_base, fixture->depth.data(), 6, amdgpu::Mtype::RW,
                                   fixture->sealed);
      } else {
        // The first block's clear changes a later key from unsupported to
        // expanded. Replaying keys gathered before those stores would throw.
        fixture->set_key(8, 0, 1);
        fixture->process.map_pages(fixture->depth_base, fixture->metadata.data(),
                                   fixture->metadata.size(), amdgpu::Mtype::RW, fixture->sealed);
      }
    }
    EXPECT_FALSE(probed.probe());
    EXPECT_FALSE(probed.materialize_leased());
    EXPECT_TRUE(probed.reporter.addresses.empty());
    if (variant < 2) {
      EXPECT_THROW(original.materialize(false), std::runtime_error);
      EXPECT_THROW(probed.materialize(true, 0x3f000000, true), std::runtime_error);
      EXPECT_FALSE(original.reporter.addresses.empty());
      EXPECT_EQ(original.key(0, 0), variant == 0 ? 0xfffc000fu : 0u);
    } else {
      EXPECT_NO_THROW(original.materialize(false, 0x3f80000f));
      EXPECT_NO_THROW(probed.materialize(true, 0x3f80000f, true));
      EXPECT_EQ(original.key(8, 0), 0x3f80000fu);
    }
    EXPECT_EQ(probed.metadata, original.metadata);
    EXPECT_EQ(probed.depth, original.depth);
    EXPECT_EQ(probed.reporter.addresses, original.reporter.addresses);
  }
}

TEST(GraphicsImageMetadataTest, ExpandedHtileProbeDeclinesFragmentedAndUnqualifiedBacking) {
  for (uint32_t variant = 0; variant < 5; ++variant) {
    SCOPED_TRACE(variant);
    ExpandedHtileFixture fixture;
    ASSERT_TRUE(fixture.access);
    auto &pte = fixture.process.page_table_.at(fixture.metadata_base >> KfdProcess::kPageShift);
    if (variant == 0) {
      pte.host_extents = {{fixture.metadata.data(), 2, 0, fixture.sealed},
                          {fixture.metadata.data() + 2, 4094, 2, fixture.sealed}};
    } else if (variant == 1) {
      pte.host_extents.push_back({fixture.metadata.data() + 4, 4, 4, fixture.sealed});
    } else if (variant == 2) {
      pte.host_extents.front().host_backed_bytes = 6;
    } else {
      pte.host_extents.front().owner = variant == 3 ? amdgpu::LegacyHostExtentOwner::Driver
                                                    : amdgpu::LegacyHostExtentOwner::Application;
    }
    const auto before = fixture.metadata;
    EXPECT_FALSE(fixture.probe());
    EXPECT_FALSE(fixture.materialize_leased());
    EXPECT_TRUE(fixture.reporter.addresses.empty());
    EXPECT_EQ(fixture.metadata, before);
    if (variant < 3) {
      EXPECT_THROW(fixture.materialize(true), std::runtime_error);
      EXPECT_FALSE(fixture.reporter.addresses.empty());
    } else {
      EXPECT_NO_THROW(fixture.materialize(true));
    }
    EXPECT_EQ(fixture.metadata, before);
  }
}

namespace {

struct ExpandedDccFixture {
  static constexpr uint64_t metadata_base = 0x100000, image_base = 0x200000;
  static constexpr uint64_t metadata_xor = 0x700, slice_size = 262144;
  static constexpr auto sealed = amdgpu::LegacyHostExtentOwner::DriverSealedRam;
  ExpandedHtileFixture::Reporter reporter;
  amdgpu::GpuMemory memory{"expanded_dcc"};
  KfdProcess process{7};
  std::vector<uint8_t> metadata = std::vector<uint8_t>(131072, 0x5a);
  std::vector<uint8_t> pixels = std::vector<uint8_t>(4 * slice_size, 0x2a);
  amdgpu::GpuVm vm;
  amdgpu::LegacyGpuVmAdapter adapter{vm, &memory};
  std::optional<amdgpu::GpuVmAccess> access;
  uint32_t width, height, bytes, swizzle, layer, bw, bh;
  bool pipe_aligned;

  ExpandedDccFixture(uint32_t width = 24, uint32_t height = 8, uint32_t bytes = 4,
                     uint32_t swizzle = 27, bool pipe_aligned = true, uint32_t layer = 0)
      : width(width), height(height), bytes(bytes), swizzle(swizzle), layer(layer),
        bw(1u << ((8 - std::countr_zero(bytes) + 1) / 2)),
        bh(1u << ((8 - std::countr_zero(bytes)) / 2)), pipe_aligned(pipe_aligned) {
    process.map_pages(metadata_base, metadata.data(), metadata.size(), amdgpu::Mtype::RW, sealed);
    process.map_pages(image_base, pixels.data(), pixels.size(), amdgpu::Mtype::RW, sealed);
    const auto handle = adapter.register_address_space(
        7, {.page_table = &process.page_table_,
            .page_table_mutex = &process.page_table_mutex_,
            .page_table_generation = process.page_table_generation(),
            .request_mutex = process.page_table_request_mutex(),
            .mutation_epoch = process.page_table_mutation_epoch(),
            .page_table_cache_state = process.page_table_cache_state(),
            .fault_reporter = &reporter});
    access = vm.snapshot(handle);
    for (uint32_t y = 0; y < height; y += bh)
      for (uint32_t x = 0; x < width; x += bw)
        set_key(x, y, 0xff);
  }
  uint64_t key_address(uint32_t x, uint32_t y) const {
    return *amdgpu::gfx11_metadata_address(metadata_base | metadata_xor, x, y, width, height, bytes,
                                           swizzle, false, pipe_aligned, layer);
  }
  void set_key(uint32_t x, uint32_t y, uint8_t key) {
    metadata.at(key_address(x, y) - metadata_base) = key;
  }
  bool probe() const {
    return amdgpu::try_gfx11_expanded_dcc(*access, metadata_base | metadata_xor, width, height,
                                          bytes, swizzle, pipe_aligned, layer);
  }
  bool materialize_leased() const {
    return amdgpu::try_materialize_gfx11_dcc_layer(*access, image_base,
                                                   metadata_base | metadata_xor, width, height,
                                                   bytes, swizzle, pipe_aligned, layer, slice_size);
  }
  void materialize(bool probe_first, bool use_lease = false) const {
    if (probe_first && probe())
      return;
    if (use_lease && materialize_leased())
      return;
    for (uint32_t y = 0; y < height; y += bh)
      for (uint32_t x = 0; x < width; x += bw)
        amdgpu::materialize_gfx11_dcc(*access, image_base, metadata_base | metadata_xor, x, y,
                                      width, height, bytes, swizzle, pipe_aligned, layer,
                                      slice_size);
  }
};

} // namespace

TEST(GraphicsImageMetadataTest, ExpandedDccProbeUsesOnlyCurrentLayerKeys) {
  for (uint32_t bytes : {1u, 2u, 4u, 8u, 16u})
    for (uint32_t swizzle : {27u, 31u})
      for (bool pipe_aligned : {false, true})
        for (uint32_t layer : {0u, 1u, 3u}) {
          SCOPED_TRACE(testing::Message()
                       << bytes << ',' << swizzle << ',' << pipe_aligned << ',' << layer);
          ExpandedDccFixture fixture(37, 19, bytes, swizzle, pipe_aligned, layer);
          ASSERT_TRUE(fixture.access);
          const auto metadata = fixture.metadata, pixels = fixture.pixels;
          EXPECT_TRUE(fixture.probe()); // Gaps and other layers still contain 0x5a.
          EXPECT_EQ(fixture.metadata, metadata);
          EXPECT_EQ(fixture.pixels, pixels);
          fixture.set_key(fixture.bw, fixture.bh, 2);
          EXPECT_FALSE(fixture.probe());
          fixture.materialize(true);
          EXPECT_TRUE(fixture.probe());
          fixture.process.unmap_pages(fixture.image_base, fixture.pixels.size());
          EXPECT_TRUE(fixture.probe());
          EXPECT_NO_THROW(fixture.materialize(false));
          EXPECT_TRUE(fixture.reporter.addresses.empty());
        }
}

TEST(GraphicsImageMetadataTest, ExpandedDccProbePreservesClearAndUnsupportedKeyPrefixes) {
  for (uint8_t key : {0, 1, 2, 3, 4, 5, 6, 8, 10, 254})
    for (uint32_t x : {0u, 8u, 16u}) {
      SCOPED_TRACE(testing::Message() << unsigned(key) << ',' << x);
      ExpandedDccFixture original, probed;
      for (auto *fixture : {&original, &probed}) {
        fixture->set_key(0, 0, 2);
        fixture->set_key(x, 0, key);
      }
      const auto before = probed.metadata;
      EXPECT_FALSE(probed.probe());
      EXPECT_EQ(probed.metadata, before);
      EXPECT_TRUE(probed.reporter.addresses.empty());
      if (key == 3 || key == 5 || key == 254) {
        EXPECT_THROW(original.materialize(false), std::runtime_error);
        EXPECT_THROW(probed.materialize(true), std::runtime_error);
      } else {
        EXPECT_NO_THROW(original.materialize(false));
        EXPECT_NO_THROW(probed.materialize(true));
      }
      EXPECT_EQ(probed.metadata, original.metadata);
      EXPECT_EQ(probed.pixels, original.pixels);
      EXPECT_EQ(probed.reporter.addresses, original.reporter.addresses);
    }
}

TEST(GraphicsImageMetadataTest, ExpandedDccProbePreservesHostStateAndDeclinesInvalidLayouts) {
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  ExpandedDccFixture fixture;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
    ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
    errno = EDOM;
    EXPECT_TRUE(fixture.probe());
    const int observed_errno = errno;
    EXPECT_EQ(observed_errno, EDOM);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_INVALID | FE_DIVBYZERO);
    EXPECT_EQ(std::fegetround(), mode);
  }
  for (uint32_t width : {0u, 4097u})
    EXPECT_FALSE(amdgpu::try_gfx11_expanded_dcc(*fixture.access, fixture.metadata_base, width, 8, 4,
                                                27, true, 0));
  for (uint32_t bytes : {0u, 3u, 32u})
    EXPECT_FALSE(amdgpu::try_gfx11_expanded_dcc(*fixture.access, fixture.metadata_base, 24, 8,
                                                bytes, 27, true, 0));
  EXPECT_FALSE(
      amdgpu::try_gfx11_expanded_dcc(*fixture.access, fixture.metadata_base, 24, 8, 4, 0, true, 0));
  EXPECT_TRUE(fixture.reporter.addresses.empty());
}

TEST(GraphicsImageMetadataTest, ExpandedDccProbePreservesFaultAndAliasedStorePrefixes) {
  for (uint32_t variant = 0; variant < 3; ++variant) {
    SCOPED_TRACE(variant);
    ExpandedDccFixture original, probed;
    for (auto *fixture : {&original, &probed}) {
      fixture->set_key(0, 0, 2);
      if (variant == 0) {
        fixture->process.unmap_pages(fixture->metadata_base, fixture->metadata.size());
        const auto first = fixture->key_address(0, 0);
        fixture->process.map_pages(first, fixture->metadata.data() + first - fixture->metadata_base,
                                   1, amdgpu::Mtype::RW, fixture->sealed);
      } else if (variant == 1) {
        fixture->process.unmap_pages(fixture->image_base, fixture->pixels.size());
        fixture->process.map_pages(fixture->image_base, fixture->pixels.data(), 6,
                                   amdgpu::Mtype::RW, fixture->sealed);
      } else {
        // The first clear replaces an unsupported later metadata key with 0xff.
        fixture->set_key(8, 0, 3);
        fixture->process.map_pages(
            fixture->image_base, fixture->metadata.data() + fixture->metadata_xor,
            fixture->metadata.size() - fixture->metadata_xor, amdgpu::Mtype::RW, fixture->sealed);
      }
    }
    EXPECT_FALSE(probed.probe());
    EXPECT_FALSE(probed.materialize_leased());
    EXPECT_TRUE(probed.reporter.addresses.empty());
    if (variant < 2) {
      EXPECT_THROW(original.materialize(false), std::runtime_error);
      EXPECT_THROW(probed.materialize(true, true), std::runtime_error);
      EXPECT_FALSE(original.reporter.addresses.empty());
    } else {
      EXPECT_NO_THROW(original.materialize(false));
      EXPECT_NO_THROW(probed.materialize(true, true));
    }
    EXPECT_EQ(probed.metadata, original.metadata);
    EXPECT_EQ(probed.pixels, original.pixels);
    EXPECT_EQ(probed.reporter.addresses, original.reporter.addresses);
  }
}

TEST(GraphicsImageMetadataTest, ExpandedDccProbeDeclinesFragmentedAndUnqualifiedBacking) {
  for (uint32_t variant = 0; variant < 5; ++variant) {
    SCOPED_TRACE(variant);
    ExpandedDccFixture fixture;
    auto &pte = fixture.process.page_table_.at(fixture.metadata_base >> KfdProcess::kPageShift);
    const uint32_t split = fixture.metadata_xor + 1;
    if (variant == 0) {
      pte.host_extents = {{fixture.metadata.data(), split, 0, fixture.sealed},
                          {fixture.metadata.data() + split, 4096 - split, split, fixture.sealed}};
    } else if (variant == 1) {
      pte.host_extents.push_back({fixture.metadata.data() + split, 1, split, fixture.sealed});
    } else if (variant == 2) {
      pte.host_extents.front().host_backed_bytes = split;
    } else {
      pte.host_extents.front().owner = variant == 3 ? amdgpu::LegacyHostExtentOwner::Driver
                                                    : amdgpu::LegacyHostExtentOwner::Application;
    }
    const auto metadata = fixture.metadata, pixels = fixture.pixels;
    EXPECT_FALSE(fixture.probe());
    EXPECT_TRUE(fixture.reporter.addresses.empty());
    EXPECT_EQ(fixture.metadata, metadata);
    EXPECT_EQ(fixture.pixels, pixels);
  }
}

namespace {

template <typename Function> std::string metadata_error(Function &&function) {
  try {
    function();
    return {};
  } catch (const std::runtime_error &error) {
    return error.what();
  }
}

} // namespace

TEST(GraphicsImageMetadataTest, LeasedDccMatchesEveryClearFormatLayerAndPartialBlock) {
  for (uint32_t bytes : {1u, 2u, 4u, 8u, 16u})
    for (uint32_t swizzle : {27u, 31u})
      for (bool pipe_aligned : {false, true})
        for (uint32_t layer : {0u, 1u, 3u})
          for (uint8_t key : {0, 1, 2, 4, 6, 8, 10, 255}) {
            SCOPED_TRACE(testing::Message() << bytes << ',' << swizzle << ',' << pipe_aligned << ','
                                            << layer << ',' << unsigned(key));
            ExpandedDccFixture original(37, 19, bytes, swizzle, pipe_aligned, layer);
            ExpandedDccFixture leased(37, 19, bytes, swizzle, pipe_aligned, layer);
            for (auto *fixture : {&original, &leased}) {
              // Keep one preceding clear, then use every remaining selected
              // key, including clipped edge blocks. Gaps and other layers stay
              // intact.
              for (uint32_t y = 0; y < fixture->height; y += fixture->bh)
                for (uint32_t x = 0; x < fixture->width; x += fixture->bw)
                  fixture->set_key(x, y, x || y ? key : 2);
              // Key 1 reads its actual pixel value, not a prefetched constant.
              const auto base = amdgpu::image_layer_base(
                  false, fixture->image_base, fixture->slice_size, layer, bytes, swizzle);
              for (uint32_t y = 0; y < fixture->height; y += fixture->bh)
                for (uint32_t x = 0; x < fixture->width; x += fixture->bw) {
                  const auto address =
                      *amdgpu::gfx11_image_address(base, x, y, fixture->width, bytes, swizzle);
                  for (uint32_t i = 0; i < bytes; ++i)
                    fixture->pixels[address - fixture->image_base + i] = 17 + x + 3 * y + i;
                }
            }
            const auto expected = metadata_error([&] { original.materialize(false); });
            bool admitted = false;
            const auto actual = metadata_error([&] { admitted = leased.materialize_leased(); });
            EXPECT_EQ(actual, expected);
            if (actual.empty()) {
              EXPECT_TRUE(admitted);
            }
            EXPECT_EQ(leased.metadata, original.metadata);
            EXPECT_EQ(leased.pixels, original.pixels);
            EXPECT_TRUE(leased.reporter.addresses.empty());
          }
}

TEST(GraphicsImageMetadataTest, LeasedMetadataPreservesLaterMalformedKeyPrefixAndReleasesGuards) {
  for (uint8_t key : {3, 5, 254}) {
    ExpandedDccFixture original, leased;
    for (auto *fixture : {&original, &leased}) {
      fixture->set_key(0, 0, 2);
      fixture->set_key(8, 0, key);
    }
    EXPECT_EQ(metadata_error([&] { leased.materialize_leased(); }),
              metadata_error([&] { original.materialize(false); }));
    EXPECT_EQ(leased.metadata, original.metadata);
    EXPECT_EQ(leased.pixels, original.pixels);
    EXPECT_TRUE(leased.reporter.addresses.empty());
    // A writer can immediately retire the mapping after the exception.
    ASSERT_TRUE(leased.process.page_table_request_mutex()->try_lock());
    leased.process.page_table_request_mutex()->unlock();
  }
  for (uint32_t key = 1; key < 15; ++key) {
    ExpandedHtileFixture original(24), leased(24);
    for (auto *fixture : {&original, &leased}) {
      fixture->set_key(0, 0, 0);
      fixture->set_key(8, 0, 0x55555550 | key);
    }
    EXPECT_EQ(metadata_error([&] { leased.materialize_leased(); }),
              metadata_error([&] { original.materialize(false); }));
    EXPECT_EQ(leased.metadata, original.metadata);
    EXPECT_EQ(leased.depth, original.depth);
    EXPECT_TRUE(leased.reporter.addresses.empty());
    ASSERT_TRUE(leased.process.page_table_request_mutex()->try_lock());
    leased.process.page_table_request_mutex()->unlock();
  }
}

TEST(GraphicsImageMetadataTest, LeasedHtilePreservesEverySharedStencilKeyBitPattern) {
  for (uint32_t swizzle : {24u, 28u})
    for (uint32_t stencil = 0; stencil <= 0x3f0; stencil += 16) {
      SCOPED_TRACE(testing::Message() << swizzle << ',' << stencil);
      ExpandedHtileFixture original(17, 13, swizzle), leased(17, 13, swizzle);
      for (auto *fixture : {&original, &leased}) {
        fixture->set_key(0, 0, 0x12340000 | stencil);
        fixture->set_key(16, 8, 0xfffffc00 | stencil);
      }
      // No stencil-pixel mapping is present or needed by depth materialization.
      ASSERT_TRUE(leased.materialize_leased(0x3f400000, true));
      original.materialize(false, 0x3f400000, false, true);
      EXPECT_EQ(leased.key(0, 0), 0xfffff00fu | stencil);
      EXPECT_EQ(leased.key(16, 8), 0xfffff00fu | stencil);
      EXPECT_EQ(leased.metadata, original.metadata);
      EXPECT_EQ(leased.depth, original.depth);
      EXPECT_TRUE(leased.reporter.addresses.empty());
    }
}

TEST(GraphicsImageMetadataTest, LeasedSharedDepthLeavesLaterStencilPhaseAndFailuresOrdered) {
  constexpr uint64_t stencil_base = 0x300000;
  for (uint32_t variant = 0; variant < 3; ++variant) {
    SCOPED_TRACE(variant);
    ExpandedHtileFixture original, leased;
    std::vector<uint8_t> original_stencil(65536, 0x5a), leased_stencil(65536, 0x5a);
    for (auto pair :
         {std::pair{&original, &original_stencil}, std::pair{&leased, &leased_stencil}}) {
      auto *fixture = pair.first;
      fixture->set_key(0, 0, 0);
      fixture->set_key(8, 0, variant == 1 ? 0x100 : 0);
      fixture->process.map_pages(stencil_base, pair.second->data(), variant == 2 ? 1 : 65536,
                                 amdgpu::Mtype::RW, fixture->sealed);
    }
    ASSERT_TRUE(leased.materialize_leased(0x3f000000, true));
    original.materialize(false, 0x3f000000, false, true);
    EXPECT_EQ(leased.metadata, original.metadata);
    EXPECT_EQ(leased.depth, original.depth);
    // A later draw can enable stencil. It still uses the ordinary separate
    // stencil loop, after all depth materialization and after the lease ends.
    const auto stencil = [&](auto &fixture) {
      for (uint32_t y = 0; y < fixture.height; y += 8)
        for (uint32_t x = 0; x < fixture.width; x += 8)
          amdgpu::materialize_gfx11_stencil_htile(*fixture.access, stencil_base,
                                                  fixture.metadata_base, x, y, fixture.width,
                                                  fixture.height, 4, fixture.swizzle, 24, 0x73);
    };
    const auto actual = metadata_error([&] { stencil(leased); });
    EXPECT_EQ(actual, metadata_error([&] { stencil(original); }));
    if (variant) {
      EXPECT_FALSE(actual.empty());
    } else {
      EXPECT_TRUE(actual.empty());
    }
    EXPECT_EQ(leased.metadata, original.metadata);
    EXPECT_EQ(leased.depth, original.depth);
    EXPECT_EQ(leased_stencil, original_stencil);
    EXPECT_EQ(leased.reporter.addresses, original.reporter.addresses);
  }
}

TEST(GraphicsImageMetadataTest, LeasedSharedDepthPreservesFaultAndAliasedKeyPrefixes) {
  for (uint32_t variant = 0; variant < 3; ++variant) {
    SCOPED_TRACE(variant);
    ExpandedHtileFixture original, leased;
    for (auto *fixture : {&original, &leased}) {
      fixture->set_key(0, 0, 0);
      if (variant == 0) {
        fixture->process.unmap_pages(fixture->metadata_base, fixture->metadata.size());
        fixture->process.map_pages(fixture->metadata_base, fixture->metadata.data(), 4,
                                   amdgpu::Mtype::RW, fixture->sealed);
      } else if (variant == 1) {
        fixture->process.unmap_pages(fixture->depth_base, fixture->depth.size());
        fixture->process.map_pages(fixture->depth_base, fixture->depth.data(), 6, amdgpu::Mtype::RW,
                                   fixture->sealed);
      } else {
        fixture->set_key(8, 0, 1);
        fixture->process.map_pages(fixture->depth_base, fixture->metadata.data(),
                                   fixture->metadata.size(), amdgpu::Mtype::RW, fixture->sealed);
      }
    }
    const auto metadata = leased.metadata, depth = leased.depth;
    EXPECT_FALSE(leased.materialize_leased(0x3f80000f, true));
    EXPECT_EQ(leased.metadata, metadata);
    EXPECT_EQ(leased.depth, depth);
    EXPECT_TRUE(leased.reporter.addresses.empty());
    const auto actual = metadata_error([&] { leased.materialize(false, 0x3f80000f, true, true); });
    EXPECT_EQ(actual,
              metadata_error([&] { original.materialize(false, 0x3f80000f, false, true); }));
    EXPECT_EQ(leased.metadata, original.metadata);
    EXPECT_EQ(leased.depth, original.depth);
    EXPECT_EQ(leased.reporter.addresses, original.reporter.addresses);
  }
}

TEST(GraphicsImageMetadataTest, LeasedHtilePreservesClearBitsPartialBlocksAndHostState) {
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (bool shared_stencil : {false, true})
    for (uint32_t swizzle : {24u, 28u})
      for (uint32_t clear : {0u, 0x80000000u, 0x3f000000u, 0x3f800000u, 0x7f800001u})
        for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
          ExpandedHtileFixture original(17, 13, swizzle), leased(17, 13, swizzle);
          for (auto *fixture : {&original, &leased})
            for (uint32_t y = 0; y < fixture->height; y += 8)
              for (uint32_t x = 0; x < fixture->width; x += 8)
                fixture->set_key(x, y, x == 8 ? 0xfffc000f : (x ? 0xfffffc00 : 0) | 0x2a0);
          ASSERT_EQ(std::fesetround(mode), 0);
          ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
          ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
          errno = EDOM;
          const bool admitted = leased.materialize_leased(clear, shared_stencil);
          const int error = errno, flags = std::fetestexcept(FE_ALL_EXCEPT);
          EXPECT_TRUE(admitted);
          EXPECT_EQ(error, EDOM);
          EXPECT_EQ(flags, FE_INVALID | FE_DIVBYZERO);
          EXPECT_EQ(std::fegetround(), mode);
          original.materialize(false, clear, false, shared_stencil);
          EXPECT_EQ(leased.metadata, original.metadata);
          EXPECT_EQ(leased.depth, original.depth);
          EXPECT_TRUE(leased.reporter.addresses.empty());
        }
}

TEST(GraphicsImageMetadataTest, LeasedDccPreservesHostStateOnSuccessAndLateError) {
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  const int saved_errno = errno;
  RestoreFenvAndErrno restore{saved, saved_errno};
  for (uint8_t key : {0, 1, 2, 3, 4, 6, 8, 10})
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      ExpandedDccFixture original, leased;
      for (auto *fixture : {&original, &leased}) {
        fixture->set_key(0, 0, 2);
        fixture->set_key(8, 0, key);
      }
      ASSERT_EQ(std::fesetround(mode), 0);
      ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
      ASSERT_EQ(std::feraiseexcept(FE_INVALID | FE_DIVBYZERO), 0);
      errno = ERANGE;
      const auto actual = metadata_error([&] { leased.materialize_leased(); });
      const int error = errno, flags = std::fetestexcept(FE_ALL_EXCEPT);
      EXPECT_EQ(error, ERANGE);
      EXPECT_EQ(flags, FE_INVALID | FE_DIVBYZERO);
      EXPECT_EQ(std::fegetround(), mode);
      EXPECT_EQ(actual, metadata_error([&] { original.materialize(false); }));
      EXPECT_EQ(leased.metadata, original.metadata);
      EXPECT_EQ(leased.pixels, original.pixels);
    }
}

TEST(GraphicsImageMetadataTest, LeasedDccRefusesInvalidFragmentedOrAliasedEnvelopesBeforeEffects) {
  for (bool pixels : {false, true})
    for (uint32_t variant = 0; variant < 7; ++variant) {
      SCOPED_TRACE(testing::Message() << pixels << ',' << variant);
      ExpandedDccFixture original, leased;
      for (auto *fixture : {&original, &leased}) {
        fixture->set_key(0, 0, 2);
        const uint64_t base = pixels ? fixture->image_base : fixture->metadata_base;
        auto &pte = fixture->process.page_table_.at(base >> KfdProcess::kPageShift);
        auto *host = pixels ? fixture->pixels.data() : fixture->metadata.data();
        const uint32_t split = pixels ? 2 : fixture->metadata_xor + 1;
        if (variant == 0) {
          pte.host_extents = {{host, split, 0, fixture->sealed},
                              {host + split, 4096 - split, split, fixture->sealed}};
        } else if (variant == 1) {
          pte.host_extents.push_back({host + split, 1, split, fixture->sealed});
        } else if (variant == 2) {
          pte.host_extents.front().host_backed_bytes = split;
        } else if (variant < 5) {
          pte.host_extents.front().owner = variant == 3
                                               ? amdgpu::LegacyHostExtentOwner::Driver
                                               : amdgpu::LegacyHostExtentOwner::Application;
        } else if (variant == 5) {
          fixture->process.unmap_pages(base, 4096);
        } else {
          // Complete contiguous host spans still refuse cross-range aliases.
          fixture->process.map_pages(fixture->image_base, fixture->metadata.data(),
                                     fixture->metadata.size(), amdgpu::Mtype::RW, fixture->sealed);
        }
      }
      const auto before_metadata = leased.metadata, before_pixels = leased.pixels;
      EXPECT_FALSE(leased.materialize_leased());
      EXPECT_EQ(leased.metadata, before_metadata);
      EXPECT_EQ(leased.pixels, before_pixels);
      EXPECT_TRUE(leased.reporter.addresses.empty());
      EXPECT_EQ(metadata_error([&] { leased.materialize(false, true); }),
                metadata_error([&] { original.materialize(false); }));
      EXPECT_EQ(leased.metadata, original.metadata);
      EXPECT_EQ(leased.pixels, original.pixels);
      EXPECT_EQ(leased.reporter.addresses, original.reporter.addresses);
    }
}

TEST(GraphicsImageMetadataTest, LeasedMetadataRefusesMalformedLayoutsAndDefaultCustomTransport) {
  ExpandedDccFixture fixture;
  fixture.set_key(0, 0, 2);
  const auto before_metadata = fixture.metadata, before_pixels = fixture.pixels;
  for (uint32_t width : {0u, 4097u}) {
    EXPECT_FALSE(amdgpu::try_materialize_gfx11_dcc_layer(*fixture.access, fixture.image_base,
                                                         fixture.metadata_base, width, 8, 4, 27,
                                                         true, 0, fixture.slice_size));
  }
  for (uint32_t bytes : {0u, 3u, 32u}) {
    EXPECT_FALSE(amdgpu::try_materialize_gfx11_dcc_layer(*fixture.access, fixture.image_base,
                                                         fixture.metadata_base, 24, 8, bytes, 27,
                                                         true, 0, fixture.slice_size));
  }
  for (uint64_t base : {UINT64_MAX - 255, UINT64_MAX - fixture.slice_size}) {
    EXPECT_FALSE(amdgpu::try_materialize_gfx11_dcc_layer(
        *fixture.access, base, fixture.metadata_base, 24, 8, 4, 27, true, 3, fixture.slice_size));
  }
  EXPECT_FALSE(amdgpu::try_materialize_gfx11_dcc_layer(*fixture.access, fixture.image_base,
                                                       UINT64_MAX - 255, 24, 8, 4, 27, true, 3,
                                                       fixture.slice_size));
  EXPECT_FALSE(amdgpu::try_materialize_gfx11_dcc_layer(*fixture.access, fixture.image_base,
                                                       fixture.image_base, 24, 8, 4, 27, true, 0,
                                                       fixture.slice_size));
  EXPECT_EQ(fixture.metadata, before_metadata);
  EXPECT_EQ(fixture.pixels, before_pixels);
  EXPECT_TRUE(fixture.reporter.addresses.empty());
  auto backing = std::make_shared<GraphicsBatchMemory>();
  amdgpu::GpuVm vm;
  const auto handle = vm.register_address_space(7, backing, backing);
  const auto access = vm.snapshot(handle);
  ASSERT_TRUE(access);
  EXPECT_FALSE(amdgpu::try_materialize_gfx11_dcc_layer(*access, 0x200000, 0x100000, 24, 8, 4, 27,
                                                       true, 0, 65536));
  EXPECT_FALSE(amdgpu::try_materialize_gfx11_htile_layer(*access, 0x200000, 0x100000, 24, 8, 24,
                                                         0x3f800000));
  EXPECT_EQ(backing->ram_lease_requests, 0u);
  EXPECT_EQ(backing->reads, 0u);
  EXPECT_EQ(backing->writes, 0u);
}

TEST(GraphicsImageMetadataTest, HtileEndpointClearsAndExplicitClearRegister) {
  amdgpu::GpuMemory memory{"htile_clear_memory"};
  amdgpu::GpuVm vm;
  const auto address_space =
      vm.register_address_space(0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory));
  const auto access = vm.snapshot(address_space);
  ASSERT_TRUE(access);
  constexpr uint64_t base = 0x200000, metadata = 0x100000;
  for (uint32_t bytes : {2u, 4u}) {
    const auto tag = *amdgpu::gfx11_metadata_address(metadata, 8, 8, 17, 17, bytes, 24, true);
    for (uint32_t key : {0u, 0xfffffff0u}) {
      memory.write32(tag, key);
      amdgpu::materialize_gfx11_htile(*access, base, metadata, 9, 9, 17, 13, bytes, 24);
      for (uint32_t y = 8; y < 13; ++y)
        for (uint32_t x = 8; x < 16; ++x) {
          const auto a = *amdgpu::gfx11_image_address(base, x, y, 17, bytes, 24);
          const uint32_t actual = bytes == 2 ? memory.read16(a) : memory.read32(a);
          EXPECT_EQ(actual, key == 0 ? 0 : bytes == 2 ? 65535u : 0x3f800000u);
        }
      EXPECT_EQ(memory.read32(tag), 0xfffc000fu);
    }
    memory.write32(tag, 0x80020000);
    EXPECT_THROW(amdgpu::materialize_gfx11_htile(*access, base, metadata, 9, 9, 17, 13, bytes, 24),
                 std::runtime_error);
    const uint32_t clear = bytes == 2 ? 32768 : 0x3f000000;
    amdgpu::materialize_gfx11_htile(*access, base, metadata, 9, 9, 17, 13, bytes, 24, clear);
    const auto a = *amdgpu::gfx11_image_address(base, 9, 9, 17, bytes, 24);
    EXPECT_EQ(bytes == 2 ? memory.read16(a) : memory.read32(a), clear);
    memory.write32(tag, 0xfffc0001);
    EXPECT_THROW(amdgpu::materialize_gfx11_htile(*access, base, metadata, 9, 9, 17, 13, bytes, 24),
                 std::runtime_error);
  }
}

TEST(GraphicsImageMetadataTest, SharedDepthStencilClearPreservesTheOtherAspect) {
  amdgpu::GpuMemory memory{"depth_stencil_clear_memory"};
  amdgpu::GpuVm vm;
  const auto id =
      vm.register_address_space(0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory));
  const auto access = vm.snapshot(id);
  ASSERT_TRUE(access);
  constexpr uint64_t depth = 0x200000, stencil = 0x300000, metadata = 0x100000;
  const auto tag = *amdgpu::gfx11_metadata_address(metadata, 9, 9, 17, 13, 4, 24, true);
  for (bool stencil_first : {false, true}) {
    memory.write32(tag, 0x80020000);
    const auto expand_stencil = [&] {
      amdgpu::materialize_gfx11_stencil_htile(*access, stencil, metadata, 9, 9, 17, 13, 4, 24, 24,
                                              0x5a);
    };
    if (stencil_first) {
      expand_stencil();
      EXPECT_EQ(memory.read32(tag), 0x800203f0u);
    }
    amdgpu::materialize_gfx11_htile(*access, depth, metadata, 9, 9, 17, 13, 4, 24, 0x3f000000,
                                    true);
    EXPECT_EQ(memory.read32(tag), stencil_first ? 0xfffff3ffu : 0xfffff00fu);
    expand_stencil();
    EXPECT_EQ(memory.read32(tag), 0xfffff3ffu);
    for (uint32_t y = 8; y < 13; ++y)
      for (uint32_t x = 8; x < 16; ++x) {
        EXPECT_EQ(memory.read32(*amdgpu::gfx11_image_address(depth, x, y, 17, 4, 24)), 0x3f000000u);
        EXPECT_EQ(memory.read8(*amdgpu::gfx11_image_address(stencil, x, y, 17, 1, 24)), 0x5au);
      }
    // A later image access must retain stencil data already expanded by DB.
    amdgpu::materialize_gfx11_htile(*access, stencil, metadata, 9, 9, 17, 13, 1, 24);
  }
}

TEST(GraphicsImageFilterTest, AnisotropicDescriptorControlsMatchPhysicalFilterCounts) {
  // FNV-1a digests of counts recovered from R32 impulse readbacks. The first
  // corpus covers every bias/PERF_MOD pair on GFX11; the independent threshold
  // corpus agrees byte-for-byte on physical GFX11 and GFX12.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t perf = 0; perf < 8; ++perf)
    for (uint32_t bias = 0; bias < 64; ++bias)
      for (uint32_t i = 0; i < 512; ++i) {
        const auto count = amdgpu::image_anisotropic_filter_count(
            amdgpu::image_footprint(1.0 + i / 32.0, 0, 0, 1, 1, 1), 16, 0, bias, perf);
        digest = (digest ^ count) * 1099511628211ull;
      }
  EXPECT_EQ(digest, 0x19dbaf1b0e5faf7cull);
  digest = 14695981039346656037ull;
  struct Control {
    uint32_t threshold, bias;
  };
  constexpr Control controls[] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0},  {5, 0},
                                  {6, 0}, {7, 0}, {0, 4}, {2, 4}, {2, 32}, {2, 63}};
  for (const auto &[threshold, bias] : controls)
    for (uint32_t i = 0; i < 4096; ++i) {
      const auto count = amdgpu::image_anisotropic_filter_count(
          amdgpu::image_footprint(1.0 + i / 256.0, 0, 0, 1, 1, 1), 16, threshold, bias, 4);
      digest = (digest ^ count) * 1099511628211ull;
    }
  EXPECT_EQ(digest, 0x435635cf362ec9fcull);
}

TEST(GraphicsImageFilterTest, AnisotropicTwoAxisCountsMatchPhysicalBoundaries) {
  struct Witness {
    float major, minor;
    uint32_t max_anisotropy, threshold, bias, perf_mod, expected;
  };
  // Physical R32 impulse counts on both GFX11 and GFX12. Non-unit minor
  // axes expose the two precision reductions; small axes expose bias ordering.
  constexpr Witness witnesses[] = {
      {2.420099f, 1.03f, 16, 2, 4, 4, 2},    {5.9045086f, 1.33f, 16, 2, 4, 4, 4},
      {17.743763f, 2.07f, 16, 2, 4, 4, 8},   {22.065083f, 2.07f, 16, 2, 4, 4, 10},
      {26.418175f, 2.07f, 16, 2, 4, 4, 12},  {30.701893f, 2.07f, 16, 2, 4, 4, 14},
      {2.015625f, 0.484375f, 4, 0, 4, 4, 4}, {2.015625f, 0.5f, 4, 0, 4, 4, 2},
      {4.125f, 0.5f, 8, 0, 4, 4, 6},         {4.125f, 0.515625f, 8, 0, 4, 4, 4},
      {8.25f, 0.5f, 16, 0, 4, 4, 10},        {8.25f, 0.515625f, 16, 0, 4, 4, 8},
      {16.5f, 1, 16, 0, 32, 7, 10},
  };
  for (const auto &witness : witnesses)
    EXPECT_EQ(amdgpu::image_anisotropic_filter_count(
                  amdgpu::image_footprint(witness.major, 0, 0, witness.minor, 1, 1),
                  witness.max_anisotropy, witness.threshold, witness.bias, witness.perf_mod),
              witness.expected)
        << witness.major << "," << witness.minor << "," << witness.max_anisotropy;
}

TEST_P(GraphicsExportTest, AnisotropicDirectionMatchesPhysicalMatricesAndSignedTies) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, 64, 16, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 16; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12
                               ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                               : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
      ASSERT_TRUE(address);
      const float channels[] = {float(std::max(int(x) - 32, 0)), float(std::max(32 - int(x), 0)),
                                float(std::max(int(y) - 8, 0)), float(std::max(8 - int(y), 0))};
      for (uint32_t c = 0; c < 4; ++c)
        memory_.write32(*address + 4 * c, std::bit_cast<uint32_t>(channels[c]));
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (63u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (15u << 14),
                                           (9u << 28) | 0xfac,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (1u << 9) | (1u << 27));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    std::array<uint32_t, 4> gradients;
    int32_t phase;
    std::array<uint32_t, 4> expected;
  };
  // Raw RGBA32F readbacks agree on physical GFX11 and GFX12. Independent
  // full-rank matrices exercise exponent alignment; signed rank-one inputs
  // distinguish positive and negative multiplication ties.
  constexpr Witness witnesses[] = {
      {{0x3ebb77fdu, 0xbf989c0du, 0x3f8c89d3u, 0xbf4c9a92u},
       -31,
       {0x3e1a0000u, 0x3e1e0000u, 0x3e4a0000u, 0x3e4e0000u}},
      {{0x3f522dc4u, 0x3fbd8ef4u, 0x3d8169d0u, 0x3eef87e8u},
       -15,
       {0x3dec0000u, 0x3df00000u, 0x3e620000u, 0x3e640000u}},
      {{0xbe1ce6acu, 0xbf01f41bu, 0x3fdeeb8au, 0xbdf3638du},
       0,
       {0x3e800000u, 0x3e800000u, 0x3c400000u, 0x3c400000u}},
      {{0x3f9519dau, 0xbfa705a3u, 0x3ebafd2bu, 0x3e82c808u},
       15,
       {0x3e2e0000u, 0x3e2c0000u, 0x3e3e0000u, 0x3e3c0000u}},
      {{0x3e403644u, 0x3ebd0003u, 0xbfd76762u, 0x3ef21fd3u},
       31,
       {0x3e7a0000u, 0x3e760000u, 0x3d880000u, 0x3d800000u}},
      {{0xbf2945d5u, 0xbfc61554u, 0x3eb195cdu, 0x3ebe1816u},
       -31,
       {0x3dd00000u, 0x3dd80000u, 0x3e660000u, 0x3e6a0000u}},
      {{0xbf436f54u, 0xbea8bed4u, 0x3fa85e58u, 0xbf898113u},
       -15,
       {0x3e560000u, 0x3e580000u, 0x3e0a0000u, 0x3e0c0000u}},
      {{0x3f95559au, 0xbe76914bu, 0xbf2d37e2u, 0x3fade987u},
       0,
       {0x3e320000u, 0x3e320000u, 0x3e380000u, 0x3e380000u}},
      {{0xbf9ec7cfu, 0x3f254c5fu, 0xbf95380bu, 0xbdfac4f5u},
       15,
       {0x3e7a0000u, 0x3e780000u, 0x3d800000u, 0x3d780000u}},
      {{0xbecd11b6u, 0xbf8f3667u, 0x3fa9d804u, 0x3f1f6a0du},
       31,
       {0x3e400000u, 0x3e3c0000u, 0x3e2c0000u, 0x3e280000u}},
      {{0x3f79806bu, 0x3f0e107fu, 0xbfb444cfu, 0x3f42b011u},
       -31,
       {0x3e760000u, 0x3e7a0000u, 0x3d680000u, 0x3d780000u}},
      {{0x3f9f622cu, 0xbf9a4822u, 0x3e80e2b9u, 0xbdb67449u},
       -15,
       {0x3e380000u, 0x3e3a0000u, 0x3e300000u, 0x3e300000u}},
      {{0x3f043c8cu, 0x3f8ee149u, 0xbfb354bau, 0xbed825efu},
       0,
       {0x3e500000u, 0x3e500000u, 0x3e140000u, 0x3e140000u}},
      {{0xbf93ae2eu, 0xbf9c6d8eu, 0xbde404d2u, 0xbf0e3d26u},
       15,
       {0x3e280000u, 0x3e260000u, 0x3e420000u, 0x3e420000u}},
      {{0xbe68e0cau, 0xbfc38c21u, 0x3f1c3262u, 0x3f34d3eau},
       31,
       {0x3d940000u, 0x3d8c0000u, 0x3e780000u, 0x3e740000u}},
      {{0x3f26eb77u, 0x3fa1be1du, 0xbf11fb53u, 0x3f9a8c91u},
       -31,
       {0x3c600000u, 0x3c800000u, 0x3e7e0000u, 0x3e810000u}},
      {{0x3fc00000u, 0x3d800000u, 0x00000000u, 0x00000000u},
       6,
       {0x3e800000u, 0x3e7e0000u, 0x3c400000u, 0x3c200000u}},
      {{0xbfc00000u, 0x3d800000u, 0x00000000u, 0x00000000u},
       6,
       {0x3e800000u, 0x3e7e0000u, 0x3c400000u, 0x3c200000u}},
      {{0x3fc00000u, 0xbd800000u, 0x00000000u, 0x00000000u},
       6,
       {0x3e800000u, 0x3e7e0000u, 0x3c400000u, 0x3c200000u}},
      {{0xbfc00000u, 0xbd800000u, 0x00000000u, 0x00000000u},
       6,
       {0x3e800000u, 0x3e7e0000u, 0x3c400000u, 0x3c200000u}},
  };
  wave_->set_exec((1u << std::size(witnesses)) - 1);
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane) {
    const auto &witness = witnesses[lane];
    const float shift = witness.phase / 8192.0f;
    const float values[] = {std::bit_cast<float>(witness.gradients[0]) / 64,
                            std::bit_cast<float>(witness.gradients[1]) / 16,
                            std::bit_cast<float>(witness.gradients[2]) / 64,
                            std::bit_cast<float>(witness.gradients[3]) / 16,
                            (32.5f + shift) / 64,
                            (8.5f + shift) / 16};
    for (uint32_t i = 0; i < std::size(values); ++i)
      wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(values[i]));
  }
  wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
  ASSERT_NO_FATAL_FAILURE(sample(28, 1));
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane)
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witnesses[lane].expected[c])
          << lane << ',' << c;
  EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
}

TEST_P(GraphicsExportTest, AnisotropicOrientationMatchesPhysicalDiagonalTies) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, 64, 16, 1, 0);
  ASSERT_TRUE(layout);
  // Separable ramps are symmetric under a V sign change. Ramps kinked along
  // both texel diagonals expose the footprint orientation.
  for (uint32_t y = 0; y < 16; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12
                               ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                               : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
      ASSERT_TRUE(address);
      const int sum = (int(x) - 32) + (int(y) - 8), difference = (int(x) - 32) - (int(y) - 8);
      const float channels[] = {float(std::max(sum, 0)), float(std::max(-sum, 0)),
                                float(std::max(difference, 0)), float(std::max(-difference, 0))};
      for (uint32_t c = 0; c < 4; ++c)
        memory_.write32(*address + 4 * c, std::bit_cast<uint32_t>(channels[c]));
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (63u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (15u << 14),
                                           (9u << 28) | 0xfac,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (1u << 9) | (1u << 27));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    std::array<uint32_t, 4> gradients;
    int32_t phase;
    std::array<uint32_t, 4> expected;
  };
  // Raw RGBA32F readbacks agree on physical GFX11 and GFX12. Screen rotations
  // near 45 degrees with a nearly axis-aligned texture footprint cover L1
  // diagonal ties, zero sign products, and antidiagonals chosen despite a
  // larger X sum component. Each rejects the gradient covariance sign, the
  // larger sum component, or both.
  constexpr Witness witnesses[] = {
      {{0x3ea79068u, 0xbf9d4219u, 0x3ea63db2u, 0x3f9d78adu},
       16,
       {0x3e810000u, 0x3e800000u, 0x3e808000u, 0x3e7f0000u}},
      {{0x3fa5f573u, 0x3f20fd9bu, 0x3fa5cdacu, 0xbf21a9fbu},
       -19,
       {0x3e7f0000u, 0x3e818000u, 0x3e810000u, 0x3e810000u}},
      {{0x403639a4u, 0xc08e0fcdu, 0x4035a22du, 0x408e1a42u},
       10,
       {0x3f808000u, 0x3f800000u, 0x3f800000u, 0x3f800000u}},
      {{0x3ebd64d5u, 0xbf960f73u, 0x3ebc4ec3u, 0x3f9655fau},
       -18,
       {0x3e800000u, 0x3e820000u, 0x3e807f00u, 0x3e807f00u}},
      {{0x3ee7436eu, 0x3f8a4a3fu, 0xbee9c552u, 0x3f897ab6u},
       18,
       {0x3e810000u, 0x3e7e0000u, 0x3e7f0000u, 0x3e7f0000u}},
      {{0x40356da6u, 0x409c53ccu, 0xc0353cf8u, 0x409cb8bbu},
       4,
       {0x3f804000u, 0x3f7f8000u, 0x3f7f8000u, 0x3f7f8000u}},
      {{0xbf22e166u, 0x3f9def9eu, 0xbf212f43u, 0xbf9f33bcu},
       -20,
       {0x3e7e0000u, 0x3e820000u, 0x3e807f00u, 0x3e7efe00u}},
      {{0x3f10097eu, 0xbf9a8bc8u, 0xbf0edfb5u, 0xbf9b3cd4u},
       -18,
       {0x3e7e0000u, 0x3e810000u, 0x3e7f0000u, 0x3e7f0000u}},
      {{0xbea160cau, 0xbf8b1e24u, 0xbea1c82cu, 0x3f8bcbc4u},
       -13,
       {0x3e800000u, 0x3e810000u, 0x3e7f0000u, 0x3e808000u}},
      {{0xbf8c39dau, 0x3f20182fu, 0x3f8be7fbu, 0x3f20aa04u},
       20,
       {0x3e818100u, 0x3e7d0200u, 0x3e810000u, 0x3e800000u}},
      {{0xbf032ea9u, 0x3f9a35b7u, 0xbf033dd3u, 0xbf99f253u},
       -14,
       {0x3e7d0200u, 0x3e808100u, 0x3e800000u, 0x3e800000u}},
      {{0xbfa4eea5u, 0x3f1ee1f0u, 0x3fa58748u, 0x3f1de46cu},
       -19,
       {0x3e7e0000u, 0x3e810000u, 0x3e7f0000u, 0x3e7f0000u}},
  };
  wave_->set_exec((1u << std::size(witnesses)) - 1);
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane) {
    const auto &witness = witnesses[lane];
    const float shift = witness.phase / 8192.0f;
    const float values[] = {std::bit_cast<float>(witness.gradients[0]) / 64,
                            std::bit_cast<float>(witness.gradients[1]) / 16,
                            std::bit_cast<float>(witness.gradients[2]) / 64,
                            std::bit_cast<float>(witness.gradients[3]) / 16,
                            (32.5f + shift) / 64,
                            (8.5f + shift) / 16};
    for (uint32_t i = 0; i < std::size(values); ++i)
      wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(values[i]));
  }
  wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
  ASSERT_NO_FATAL_FAILURE(sample(28, 1));
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane)
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witnesses[lane].expected[c])
          << lane << ',' << c;
  EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
}

TEST_P(GraphicsExportTest, AnisotropicCoordinatesMatchPhysicalExtentsAndAlignment) {
  struct Witness {
    uint32_t width, height;
    bool repeat;
    std::array<uint32_t, 6> values;
    std::array<uint32_t, 4> expected;
  };
  // Raw gradient, coordinate and RGBA32F bits captured on GFX11 and GFX12.
  // The controls cover exact-power norm reciprocals, 19x13 and 29x31 image
  // reciprocals, and positive/negative coordinates near zero and repeat edges.
  constexpr Witness witnesses[] = {
      {64,
       16,
       false,
       {0x3cd59b3du, 0xbca87c86u, 0x3c196de4u, 0x3d34550bu, 0x3f01fc80u, 0x3f07f200u},
       {0x3e7e0000u, 0x3e810000u, 0x3c400000u, 0x3c600000u}},
      {64,
       16,
       false,
       {0x3cd59b3du, 0xbca87c86u, 0x3c196de4u, 0x3d34550bu, 0x3f0200a0u, 0x3f080280u},
       {0x3e800000u, 0x3e800000u, 0x3c600000u, 0x3c400000u}},
      {19,
       13,
       false,
       {0xba537364u, 0x3cbe5c60u, 0xbdbb2f03u, 0x3cb61294u, 0x3effe86cu, 0x3effdd8au},
       {0x3e7a0000u, 0x3e7e0000u, 0x3d280000u, 0x3d380000u}},
      {19,
       13,
       false,
       {0xbd591efdu, 0xbcc709eeu, 0x3d6b9cfeu, 0xbdb019e3u, 0x3effe6bdu, 0x3effdb14u},
       {0x3e560000u, 0x3e5a0000u, 0x3e080000u, 0x3e0a0000u}},
      {29,
       31,
       false,
       {0x3d76f05fu, 0x3c463a5bu, 0xbc8d43d3u, 0x3c3ac217u, 0x3effee58u, 0x3effef7bu},
       {0x3e7a0000u, 0x3e7e0000u, 0x3d100000u, 0x3d200000u}},
      {29,
       31,
       false,
       {0x3d6bb420u, 0xbc2debe5u, 0x3ca94d1fu, 0x3cba263du, 0x3f00088du, 0x3f000800u},
       {0x3e810000u, 0x3e7e0000u, 0x3c600000u, 0x3c200000u}},
      {19,
       13,
       true,
       {0xbccb6817u, 0xbe0978ceu, 0x3a3d3244u, 0xbd096a13u, 0xbeffe5e5u, 0xbeffd9d8u},
       {0x3d800000u, 0x3d780000u, 0x3e7a0000u, 0x3e760000u}},
      {19,
       13,
       true,
       {0xbcc1e249u, 0xbde508a5u, 0x3d09b8a9u, 0xbd61e869u, 0xbefffe51u, 0xbefffd89u},
       {0x3cc00000u, 0x3cc00000u, 0x3e800000u, 0x3e800000u}},
      {19,
       13,
       true,
       {0x3dbc7413u, 0x3cec5947u, 0xbcd79d6au, 0x3cdeac58u, 0x3fbff943u, 0x3fbff628u},
       {0x3e7a0000u, 0x3e7e0000u, 0x3d100000u, 0x3d200000u}},
      {19,
       13,
       true,
       {0x3db3e10bu, 0xbccf5e2fu, 0x3d01341eu, 0x3d5df284u, 0x3fc00687u, 0x3fc0098au},
       {0x3e810000u, 0x3e7e0000u, 0x3c600000u, 0x3c200000u}},
      {19,
       13,
       true,
       {0x3d71969bu, 0x3d8f53d7u, 0xbd9169ccu, 0x3cb01940u, 0x30820000u, 0x30820000u},
       {0x40900000u, 0x40900000u, 0x40400000u, 0x40400000u}},
      {19,
       13,
       true,
       {0x3d054cc6u, 0x3b07f324u, 0xbd5f5b14u, 0x3dd9ca9cu, 0x30a00000u, 0x30a00000u},
       {0x40900000u, 0x40900000u, 0x40400000u, 0x40400000u}},
      {19,
       13,
       true,
       {0x3d71969bu, 0x3d8f53d7u, 0xbd9169ccu, 0x3cb01940u, 0xb0820000u, 0xb0820000u},
       {0x40900000u, 0x40900000u, 0x40400000u, 0x40400000u}},
      {19,
       13,
       true,
       {0x3d054cc6u, 0x3b07f324u, 0xbd5f5b14u, 0x3dd9ca9cu, 0xb0a00000u, 0xb0a00000u},
       {0x40900000u, 0x40900000u, 0x40400000u, 0x40400000u}},
  };
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  wave_->set_exec(1);
  for (const auto &witness : witnesses) {
    SCOPED_TRACE(::testing::Message()
                 << witness.width << ',' << witness.height << ',' << std::hex << witness.values[4]);
    const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, witness.width, witness.height, 1, 0);
    ASSERT_TRUE(layout);
    for (uint32_t y = 0; y < witness.height; ++y)
      for (uint32_t x = 0; x < witness.width; ++x) {
        const auto address =
            gfx12 ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                  : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
        ASSERT_TRUE(address);
        const int dx = int(x) - int(witness.width / 2);
        const int dy = int(y) - int(witness.height / 2);
        const float channels[] = {float(std::max(dx, 0)), float(std::max(-dx, 0)),
                                  float(std::max(dy, 0)), float(std::max(-dy, 0))};
        for (uint32_t c = 0; c < 4; ++c)
          memory_.write32(*address + 4 * c, std::bit_cast<uint32_t>(channels[c]));
      }
    const uint32_t width = witness.width - 1;
    const std::array<uint32_t, 8> descriptor{0x1000,
                                             (63u << (gfx12 ? 17 : 20)) | ((width & 3) << 30),
                                             (width >> 2) | ((witness.height - 1) << 14),
                                             (9u << 28) | 0xfac,
                                             0,
                                             4u << 20,
                                             0,
                                             0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, (witness.repeat ? 0 : 0x92) | (1u << 9) | (1u << 27));
    wave_->debug_write_sgpr(5, 0);
    wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
    wave_->debug_write_sgpr(7, 0);
    for (uint32_t i = 0; i < witness.values.size(); ++i)
      wave_->debug_write_vgpr(i, 0, witness.values[i]);
    ASSERT_NO_FATAL_FAILURE(sample(28, 1));
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), witness.expected[c]) << c;
  }
}

TEST(GraphicsImageFilterTest, AnisotropicNormalizationMatchesAllPhysicalMantissas) {
  // Direction components inferred independently from cardinal phase captures
  // below and above two texels. Compare the complete finite mantissa domain.
  uint64_t digest = 14695981039346656037ull;
  for (uint32_t i = 0; i < 1024; ++i) {
    const double gradient = (1.0 + i / 1024.0) / 64;
    const auto direction = amdgpu::image_footprint(gradient, 0, 0, 0, 64, 16).direction();
    EXPECT_EQ(direction[1], 0);
    digest = (digest ^ std::bit_cast<uint64_t>(direction[0])) * 1099511628211ull;
  }
  EXPECT_EQ(digest, 0x282fbe27df287325ull);
}

TEST_P(GraphicsExportTest, AnisotropicSamplingUsesPhysicalNonuniformWeights) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 4, 64, 16, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 16; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12 ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 4, 0)
                                 : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 4, 0);
      ASSERT_TRUE(address);
      memory_.write32(*address, x == 32 ? 0x3f800000 : 0);
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (22u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (15u << 14),
                                           (9u << 28) | 0x204,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (4u << 9) | (2u << 16) | (4u << 21) | (1u << 27));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (2u << 20) | (2u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    uint32_t count;
    std::array<uint32_t, 14> expected;
  };
  // Raw R32 impulse outputs for nearest anisotropic filtering on both cards.
  constexpr Witness witnesses[] = {
      {6, {0x3e2a0000, 0x3e2b0000, 0x3e2b0000, 0x3e2b0000, 0x3e2b0000, 0x3e2a0000}},
      {10,
       {0x3dcc0000, 0x3dcd0000, 0x3dcd0000, 0x3dcd0000, 0x3dcd0000, 0x3dcd0000, 0x3dcd0000,
        0x3dcd0000, 0x3dcd0000, 0x3dcc0000}},
      {12,
       {0x3daa0000, 0x3daa0000, 0x3dab0000, 0x3dab0000, 0x3dab0000, 0x3dab0000, 0x3dab0000,
        0x3dab0000, 0x3dab0000, 0x3dab0000, 0x3daa0000, 0x3daa0000}},
      {14,
       {0x3d920000, 0x3d920000, 0x3d920000, 0x3d920000, 0x3d920000, 0x3d930000, 0x3d930000,
        0x3d930000, 0x3d930000, 0x3d920000, 0x3d920000, 0x3d920000, 0x3d920000, 0x3d920000}},
  };
  for (const auto &witness : witnesses) {
    wave_->set_exec((1u << witness.count) - 1);
    for (uint32_t lane = 0; lane < witness.count; ++lane) {
      const float offset = float(lane) + 0.125f - witness.count * 0.5f;
      const float values[] = {witness.count / 64.0f, 0,        0, 1.0f / 16,
                              (32.5f + offset) / 64, 8.5f / 16};
      for (uint32_t i = 0; i < std::size(values); ++i)
        wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(values[i]));
    }
    wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
    ASSERT_NO_FATAL_FAILURE(sample(28, 1)); // IMAGE_SAMPLE_D, 2D.
    for (uint32_t lane = 0; lane < witness.count; ++lane) {
      EXPECT_EQ(wave_->debug_read_vgpr(12, lane), witness.expected[lane])
          << witness.count << "," << lane;
      EXPECT_EQ(wave_->debug_read_vgpr(13, lane), 0u);
      EXPECT_EQ(wave_->debug_read_vgpr(14, lane), 0u);
      EXPECT_EQ(wave_->debug_read_vgpr(15, lane), 0x3f800000u);
    }
    EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
  }
  // A footprint above the sampler's maximum still applies bias before the
  // count cap. Both cards select ten filters here; pre-capping selects eight.
  wave_->debug_write_sgpr(13, 7u << 20);
  wave_->debug_write_sgpr(4, 0x92 | (4u << 9) | (32u << 21) | (1u << 27));
  wave_->set_exec(1);
  const float values[] = {16.5f / 64, 0, 0, 1.0f / 16, 33.75f / 64, 8.5f / 16};
  for (uint32_t i = 0; i < std::size(values); ++i)
    wave_->debug_write_vgpr(i, 0, std::bit_cast<uint32_t>(values[i]));
  ASSERT_NO_FATAL_FAILURE(sample(28, 1));
  EXPECT_EQ(wave_->debug_read_vgpr(12, 0), 0x3dcd0000u);
}

TEST_P(GraphicsExportTest, AnisotropicArraySamplingUsesIndependentLaneFootprints) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 4, 64, 16, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 16; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      uint32_t value = x * 1664525u + y * 1013904223u;
      value ^= value >> 16;
      value = value * 2246822519u | 0xff000000u;
      const auto address = gfx12 ? amdgpu::gfx12_image_address(0x100000 + 2 * layout->slice_size, x,
                                                               y, layout->pitch, 4, 0)
                                 : amdgpu::gfx11_image_address(0x100000 + 2 * layout->slice_size, x,
                                                               y, layout->pitch, 4, 0);
      ASSERT_TRUE(address);
      memory_.write32(*address, value);
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (42u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (15u << 14),
                                           (13u << 28) | 0xfac,
                                           (2u << 16) | 2,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (4u << 9) | (2u << 16) | (4u << 21));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Footprint {
    float major, minor, u, v;
  };
  struct Witness {
    Footprint footprint;
    std::array<uint32_t, 4> expected;
  };
  // Exact raw outputs from 64x16 RGBA8 image samples on both physical targets.
  const Witness witnesses[] = {
      {{1.0f, 0.5f, 0.537109375f, 0.37f}, {0x3e7c3030u, 0x3f197dfeu, 0x3e8f1515u, 0x3f800000u}},
      {{2.0f, 0.5f, 0.537109375f, 0.37f}, {0x3e999091u, 0x3f019fe0u, 0x3ea60000u, 0x3f800000u}},
      {{4.0f, 0.5f, 0.537109375f, 0.37f}, {0x3eb72f2fu, 0x3ecd8505u, 0x3ee05adbu, 0x3f800000u}},
      {{8.0f, 0.5f, 0.537109375f, 0.37f}, {0x3eca9e5eu, 0x3ec755b6u, 0x3ec3de9fu, 0x3f800000u}},
      {{16.0f, 0.5f, 0.537109375f, 0.37f}, {0x3ee2acadu, 0x3ef34c2cu, 0x3ecb8707u, 0x3f800000u}},
      {{1.0f, 1.0f, 0.537109375f, 0.37f}, {0x3e7c3030u, 0x3f197dfeu, 0x3e8f1515u, 0x3f800000u}},
      {{2.0f, 1.0f, 0.537109375f, 0.37f}, {0x3e999091u, 0x3f019fe0u, 0x3ea60000u, 0x3f800000u}},
      {{4.0f, 1.0f, 0.537109375f, 0.37f}, {0x3eb72f2fu, 0x3ecd8505u, 0x3ee05adbu, 0x3f800000u}},
      {{8.0f, 1.0f, 0.537109375f, 0.37f}, {0x3eca9e5eu, 0x3ec755b6u, 0x3ec3de9fu, 0x3f800000u}},
      {{6.0f, 1.0f, 0.537109375f, 0.37f}, {0x3ec2abdcu, 0x3eb81151u, 0x3ed71adbu, 0x3f800000u}},
      {{16.0f, 1.0f, 0.537109375f, 0.37f}, {0x3ee2acadu, 0x3ef34c2cu, 0x3ecb8707u, 0x3f800000u}},
      {{1.0f, 2.0f, 0.537109375f, 0.37f}, {0x3e80b8b9u, 0x3f1c46c7u, 0x3e9e9515u, 0x3f800000u}},
      {{2.0f, 2.0f, 0.537109375f, 0.37f}, {0x3e7c3030u, 0x3f197dfeu, 0x3e8f1515u, 0x3f800000u}},
      {{4.0f, 2.0f, 0.537109375f, 0.37f}, {0x3ec3b9bau, 0x3ec38081u, 0x3eda5f5fu, 0x3f800000u}},
      {{8.0f, 2.0f, 0.537109375f, 0.37f}, {0x3ecbf373u, 0x3ead7afbu, 0x3ea282c3u, 0x3f800000u}},
      {{16.0f, 2.0f, 0.537109375f, 0.37f}, {0x3ee7f373u, 0x3eed5959u, 0x3eb10525u, 0x3f800000u}},
      // Additional phase sweeps isolate anisotropic UNORM accumulation and
      // normalization rounding for power-of-two and uneven filter counts.
      {{2.0f, 1.0f, 0.00121093751f, 0.0319140628f},
       {0x3ca9f3f4u, 0x3ca44040u, 0x3d1e1495u, 0x3f800000u}},
      {{4.0f, 1.0f, 0.00121093751f, 0.0319140628f},
       {0x3dc22c6cu, 0x3d9a6cedu, 0x3e45e3a4u, 0x3f800000u}},
      {{6.0f, 1.0f, 0.415273428f, 0.0319140628f},
       {0x3e9744b5u, 0x3eff4464u, 0x3ed18919u, 0x3f800000u}},
      {{6.0f, 1.0f, 0.458242178f, 0.0397265628f},
       {0x3e9894e5u, 0x3eb67737u, 0x3ed33e2eu, 0x3f800000u}},
      {{8.0f, 1.0f, 0.0168359373f, 0.0319140628f},
       {0x3e95eddeu, 0x3e98fd7du, 0x3e9a1e9fu, 0x3f800000u}},
      {{14.0f, 1.0f, 0.0441796891f, 0.0319140628f},
       {0x3eb64aabu, 0x3ec83d3du, 0x3eb293c4u, 0x3f800000u}},
      {{14.0f, 1.0f, 0.219960943f, 0.0319140628f},
       {0x3f12ca22u, 0x3f166d55u, 0x3f0ebe0eu, 0x3f800000u}},
      {{16.0f, 1.0f, 0.376210928f, 0.0319140628f},
       {0x3ed37e5eu, 0x3f04e8c9u, 0x3f059921u, 0x3f800000u}},
  };
  wave_->set_exec((1u << std::size(witnesses)) - 1);
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane) {
    const auto &footprint = witnesses[lane].footprint;
    const float values[] = {footprint.major / 64, 0,           0, footprint.minor / 16,
                            footprint.u,          footprint.v, 0};
    for (uint32_t i = 0; i < std::size(values); ++i)
      wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(values[i]));
  }
  for (uint32_t c = 0; c < 4; ++c)
    wave_->debug_write_vgpr(12 + c, 31, 0xdeadbeef);
  ASSERT_NO_FATAL_FAILURE(sample(28, 5)); // IMAGE_SAMPLE_D, 2D array.
  for (uint32_t lane = 0; lane < std::size(witnesses); ++lane)
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witnesses[lane].expected[c])
          << lane << "," << c;
  for (uint32_t c = 0; c < 4; ++c)
    EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 31), 0xdeadbeefu);
}

TEST_P(GraphicsExportTest, CubeGatherCornersMatchPhysicalFloatAndIntegerResults) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Case {
    uint32_t lane;
    std::array<uint32_t, 4> expected, integer_expected;
  };
  // Corner and interior samples from two cubes in a view starting at face six.
  const Case cases[] = {
      {0,
       {0x447ac000u, 0x44160000u, 0x444bc000u, 0x44497fccu},
       {0x900003ebu, 0x90000258u, 0x9000032fu, 0x00000000u}},
      {3,
       {0x4461c000u, 0x44190000u, 0x4452d564u, 0x447dc000u},
       {0x90000387u, 0x90000264u, 0x00000000u, 0x900003f7u}},
      {7,
       {0x44a36000u, 0x44c88000u, 0x44c80000u, 0x44a2e000u},
       {0x9000051bu, 0x90000644u, 0x90000640u, 0x90000517u}},
      {12,
       {0x44b46a8eu, 0x44bbe000u, 0x44978000u, 0x44c9e000u},
       {0x00000000u, 0x900005dfu, 0x900004bcu, 0x9000064fu}},
      {15,
       {0x44d60000u, 0x44b9155au, 0x4497e000u, 0x44bd6000u},
       {0x900006b0u, 0x00000000u, 0x900004bfu, 0x900005ebu}},
      {21,
       {0x448a0000u, 0x448a2000u, 0x4489a000u, 0x44898000u},
       {0x90000450u, 0x90000451u, 0x9000044du, 0x9000044cu}},
  };
  for (bool integer : {false, true}) {
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    for (uint32_t layer = 0; layer < 18; ++layer)
      for (uint32_t y = 0; y < 4; ++y)
        for (uint32_t x = 0; x < 4; ++x) {
          const uint32_t value = layer * 100 + y * 4 + x;
          memory_.write32(0x100000 + layer * 1024 + y * (gfx12 ? 128 : 256) + x * 4,
                          integer ? 0x90000000u + value : std::bit_cast<uint32_t>(float(value)));
        }
    const std::array<uint32_t, 8> descriptor{0x1000,
                                             ((integer ? 20u : 22u) << (gfx12 ? 17 : 20)) |
                                                 (3u << 30),
                                             3u << 14,
                                             (11u << 28) | 0xfac,
                                             (6u << 16) | 17,
                                             0,
                                             0,
                                             0};
    for (uint32_t i = 0; i < 8; ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, 2 | (2 << 3));
    wave_->debug_write_sgpr(5, 0);
    wave_->debug_write_sgpr(6, 1u << 26);
    wave_->debug_write_sgpr(7, 0);
    for (bool a16 : {false, true})
      for (uint32_t filters : {0u, 5u << 20, 1u << 26}) {
        wave_->debug_write_sgpr(6, (1u << 26) + filters);
        wave_->set_exec((1u << std::size(cases)) - 1);
        for (uint32_t i = 0; i < std::size(cases); ++i) {
          const auto lane = cases[i].lane;
          const float face = lane < 6     ? float(lane)
                             : lane == 7  ? 9
                             : lane == 12 ? 16
                             : lane == 15 ? 19
                                          : -1;
          const float u = 0.95f + float(lane % 4) * 0.35f;
          const float v = 0.95f + float((lane / 4) % 4) * 0.35f;
          if (a16) {
            wave_->debug_write_vgpr(0, i,
                                    util::f32_to_f16(u) | (uint32_t(util::f32_to_f16(v)) << 16));
            wave_->debug_write_vgpr(1, i, util::f32_to_f16(face));
          } else {
            wave_->debug_write_vgpr(0, i, std::bit_cast<uint32_t>(u));
            wave_->debug_write_vgpr(1, i, std::bit_cast<uint32_t>(v));
            wave_->debug_write_vgpr(2, i, std::bit_cast<uint32_t>(face));
            wave_->debug_write_vgpr(3, i, 0);
          }
        }
        ASSERT_NO_FATAL_FAILURE(sample(48, 3, a16, 1));
        for (uint32_t i = 0; i < std::size(cases); ++i)
          for (uint32_t c = 0; c < 4; ++c) {
            const uint32_t expected = integer ? cases[i].integer_expected[c] : cases[i].expected[c];
            EXPECT_EQ(wave_->debug_read_vgpr(12 + c, i), expected) << i << ',' << c << ',' << a16;
          }
      }
  }
}

TEST_P(GraphicsExportTest, CubeSamplingRemapsEdgesCornersAndArrayViewFaces) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 8, 4, 4, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t face = 0; face < 12; ++face)
    for (uint32_t y = 0; y < 4; ++y)
      for (uint32_t x = 0; x < 4; ++x) {
        const auto address =
            gfx12 ? amdgpu::gfx12_image_address(0x100000 + (6 + face) * layout->slice_size, x, y,
                                                layout->pitch, 8, 0)
                  : amdgpu::gfx11_image_address(0x100000 + (6 + face) * layout->slice_size, x, y,
                                                layout->pitch, 8, 0);
        ASSERT_TRUE(address);
        for (uint32_t c = 0; c < 4; ++c)
          memory_.write16(*address + 2 * c, 0x2800 + (face % 6) * 0x500 + (face / 6) * 0x400 +
                                                x * 0x21 + y * 0x31 + c * 0xb);
      }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (57u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           3u << 14,
                                           (11u << 28) | 0xfac,
                                           (6u << 16) | 17,
                                           0,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92);
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (1u << 20) | (1u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    uint32_t query;
    std::array<uint32_t, 4> expected;
  };
  // Face-edge and corner outputs captured from a 4x4 RGBA16F cube on both cards.
  const Witness witnesses[] = {
      {0, {0x3ee3e5e0u, 0x3ee5f93cu, 0x3ee81d19u, 0x3eea30f8u}},
      {15, {0x3e955181u, 0x3e9694d6u, 0x3e97f04au, 0x3e99345du}},
      {16, {0x3f0c7f20u, 0x3f0d815au, 0x3f0e9c34u, 0x3f0fceeeu}},
      {31, {0x3e14e50cu, 0x3e15804eu, 0x3e169c8bu, 0x3e1835ccu}},
      {32, {0x3fb5d6e3u, 0x3fb741a3u, 0x3fb8ac63u, 0x3fba2763u}},
      {47, {0x3fb4f27du, 0x3fb65d3du, 0x3fb7c7fdu, 0x3fb942fdu}},
      {48, {0x3f4a4587u, 0x3f4bf578u, 0x3f4dad47u, 0x3f4f5d38u}},
      {63, {0x3f4b7619u, 0x3f4d29d8u, 0x3f4eddd9u, 0x3f509198u}},
      {64, {0x3f1e3f9cu, 0x3f1fcc5du, 0x3f21611du, 0x3f22f5ddu}},
      {79, {0x3f1f5c34u, 0x3f20e543u, 0x3f227a03u, 0x3f240ec3u}},
      {80, {0x3fb20a03u, 0x3fb37f43u, 0x3fb50443u, 0x3fb67983u}},
      {95, {0x3fb2f69du, 0x3fb46bddu, 0x3fb5f0ddu, 0x3fb7661du}},
      {255, {0x3f14048eu, 0x3f152edcu, 0x3f1658e9u, 0x3f178b37u}},
      {256, {0x3ee03546u, 0x3ee26d36u, 0x3ee4b5a4u, 0x3ee6ed95u}},
      {32767, {0x3f127baeu, 0x3f13ee6eu, 0x3f15616eu, 0x3f16cc2eu}},
      {32768, {0x3e0f1870u, 0x3e1021d0u, 0x3e112b30u, 0x3e123490u}},
      {65280, {0x3f13270cu, 0x3f144fd4u, 0x3f1578ddu, 0x3f16a9e5u}},
      {65295, {0x3ed5f1abu, 0x3ed78295u, 0x3ed9143cu, 0x3edabd45u}},
      {65311, {0x3e94fa4du, 0x3e95c52cu, 0x3e96d088u, 0x3e97dbe4u}},
      {65327, {0x3f19c82cu, 0x3f1b4f7du, 0x3f1cdabdu, 0x3f1e65fdu}},
      {65343, {0x3fd8c65au, 0x3fda499bu, 0x3fdbd4dbu, 0x3fdd601bu}},
      {65359, {0x3f4fa087u, 0x3f516078u, 0x3f5318c7u, 0x3f54d8b8u}},
      {65520, {0x3facee23u, 0x3fae39fau, 0x3fafa932u, 0x3fb10c5au}},
      {65535, {0x3f8d244eu, 0x3f8e3680u, 0x3f8f5febu, 0x3f908115u}},
  };
  for (const auto &[opcode, name] :
       {std::pair{31u, "IMAGE_SAMPLE_LZ"}, {57u, "IMAGE_SAMPLE_D_G16"}})
    for (bool a16 : {false, true})
      for (uint32_t cube : {0u, 1u}) {
        SCOPED_TRACE(testing::Message() << "opcode=" << name << ", a16=" << a16);
        const uint32_t prefix = opcode == 57 ? 2 : 0;
        wave_->set_exec((1u << std::size(witnesses)) - 1);
        for (uint32_t lane = 0; lane < std::size(witnesses); ++lane) {
          const uint32_t q = witnesses[lane].query;
          const float values[] = {1 + ((q & 255) + 0.5f) / 256, 1 + ((q >> 8) + 0.5f) / 256,
                                  float((q >> 4) % 6 + cube * 8)};
          for (uint32_t i = 0; i < prefix; ++i)
            wave_->debug_write_vgpr(i, lane, 0);
          if (a16) {
            wave_->debug_write_vgpr(prefix, lane,
                                    util::f32_to_f16(values[0]) |
                                        (uint32_t(util::f32_to_f16(values[1])) << 16));
            wave_->debug_write_vgpr(prefix + 1, lane, util::f32_to_f16(values[2]));
          } else {
            for (uint32_t i = 0; i < 3; ++i)
              wave_->debug_write_vgpr(prefix + i, lane, std::bit_cast<uint32_t>(values[i]));
          }
        }
        wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
        ASSERT_NO_FATAL_FAILURE(sample(opcode, 3, a16));
        for (uint32_t lane = 0; lane < std::size(witnesses); ++lane)
          for (uint32_t c = 0; c < 4; ++c)
            EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane),
                      witnesses[lane].expected[c] + (cube << 23))
                << lane << "," << c;
        EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
      }
}

TEST_P(GraphicsExportTest, ImplicitCubeSamplingUnfoldsAdjacentFacesAndBoundsOppositeFaces) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  for (uint32_t level = 0; level < 3; ++level) {
    const auto mip = amdgpu::image_mip_layout(gfx12, 0, 8, 4, 4, 3, level);
    ASSERT_TRUE(mip);
    for (uint32_t face = 0; face < 6; ++face)
      for (uint32_t y = 0; y < mip->height; ++y)
        for (uint32_t x = 0; x < mip->width; ++x) {
          const uint64_t base = 0x100000 + mip->offset + face * mip->slice_size;
          const auto address = gfx12 ? amdgpu::gfx12_image_address(base, x, y, mip->pitch, 8, 0)
                                     : amdgpu::gfx11_image_address(base, x, y, mip->pitch, 8, 0);
          ASSERT_TRUE(address);
          for (uint32_t c = 0; c < 4; ++c)
            memory_.write16(*address + 2 * c, 0x3800 + level * 0x400 + face * 0x20);
        }
  }
  const std::array<uint32_t, 8> descriptor{
      0x1000,   (57u << (gfx12 ? 17 : 20)) | (3u << 30) | (2u << (gfx12 ? 12 : 16)),
      3u << 14, (11u << 28) | 0xfac | (2u << (gfx12 ? 15 : 16)),
      5,        0,
      0,        0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92);
  wave_->debug_write_sgpr(5, 512u << (gfx12 ? 13 : 12));
  wave_->debug_write_sgpr(6, 1u << 26);
  wave_->debug_write_sgpr(7, 0);
  // Physical RDNA3/4 captures distinguish each mip (0.5, 1, 2) and each face.
  // Include perpendicular and opposite faces, identical directions, and an edge.
  for (uint32_t opcode : {27u, 31u}) {
    struct Case {
      const char *name;
      float face, u, expected;
    };
    const Case cases[] = {{"perpendicular Y", 2.0f, 1.5f, 2.0f},
                          {"opposite X", 1.0f, 1.5f, 2.0f},
                          {"perpendicular Z", 4.0f, 1.5f, 2.0f},
                          {"identical directions", 0.0f, 1.5f, 0.5f},
                          {"adjacent Y toward shared edge", 2.0f, 1.75f, 2.0f},
                          {"adjacent Y away from shared edge", 2.0f, 1.25f, 2.0f},
                          {"shared edge", 2.0f, 2.0f, 1.0f}};
    for (const auto &[name, face, u, expected] : cases) {
      SCOPED_TRACE(opcode == 27 ? "IMAGE_SAMPLE" : "IMAGE_SAMPLE_LZ");
      SCOPED_TRACE(name);
      wave_->set_exec(15);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        wave_->debug_write_vgpr(0, lane, std::bit_cast<uint32_t>(lane & 1 ? u : 1.5f));
        wave_->debug_write_vgpr(1, lane, std::bit_cast<uint32_t>(1.5f));
        wave_->debug_write_vgpr(2, lane, std::bit_cast<uint32_t>(lane & 1 ? face : 0.0f));
      }
      ASSERT_NO_FATAL_FAILURE(sample(opcode, 3));
      // A 0x20 FP16 significand step adds 1/32 of these power-of-two mip colors.
      for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t c = 0; c < 4; ++c)
          EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane),
                    std::bit_cast<uint32_t>((opcode == 31 ? 0.5f : expected) *
                                            (lane & 1 ? 1.0f + face / 32 : 1.0f)));
    }
  }
}

TEST_P(GraphicsExportTest, CubeSamplingMatchesPhysicalFootprints) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const uint16_t colors[] = {0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600};
  const auto setup_cube = [&](uint32_t size) {
    // Keep the fixtures at separate addresses so previous texture-cache entries
    // cannot observe the direct backing-memory writes for the next image.
    const uint64_t image_base = size == 64 ? 0x100000 : 0x200000;
    for (uint32_t level = 0; level < std::size(colors); ++level) {
      const auto mip = amdgpu::image_mip_layout(gfx12, 0, 8, size, size, 7, level);
      ASSERT_TRUE(mip);
      for (uint32_t face = 0; face < 6; ++face)
        for (uint32_t y = 0; y < mip->height; ++y)
          for (uint32_t x = 0; x < mip->width; ++x) {
            const uint64_t base = image_base + mip->offset + face * mip->slice_size;
            const auto address = gfx12 ? amdgpu::gfx12_image_address(base, x, y, mip->pitch, 8, 0)
                                       : amdgpu::gfx11_image_address(base, x, y, mip->pitch, 8, 0);
            ASSERT_TRUE(address);
            for (uint32_t c = 0; c < 4; ++c)
              memory_.write16(*address + 2 * c, colors[level]);
          }
    }
    const std::array<uint32_t, 8> descriptor{
        static_cast<uint32_t>(image_base >> 8),
        (57u << (gfx12 ? 17 : 20)) | (((size - 1) & 3u) << 30) | (6u << (gfx12 ? 12 : 16)),
        ((size - 1) >> 2) | ((size - 1) << 14),
        (11u << 28) | 0xfac | (6u << (gfx12 ? 15 : 16)),
        5,
        4u << 20,
        0,
        0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, 0x92 | (4u << 9) | (2u << 16) | (4u << 21));
    wave_->debug_write_sgpr(5, (1536u << (gfx12 ? 13 : 12)) | (gfx12 ? 0 : 10u << 24));
    wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22) | (2u << 26) | (gfx12 ? 2u << 30 : 0));
    wave_->debug_write_sgpr(7, gfx12 ? 2 : 0);
  };
  setup_cube(64);
  struct Witness {
    float dxu, dxv, dyu, dyv;
    uint32_t expected;
  };
  // Both physical cards return these mip colors with 16x anisotropy enabled.
  // The gradients describe cube directions (1, -v, -u); projected coordinates
  // scale them by one half. Cover narrow, square, degenerate and rotated quads.
  const Witness witnesses[] = {
      {0.5f, 0, 0, 0.03125f, 0x40800000u},
      {0.25f, 0, 0, 0.0625f, 0x40400000u},
      {0.0625f, 0, 0, 0.5f, 0x40800000u},
      {0.0625f, 0, 0, 0.0625f, 0x3f800000u},
      {0.03125f, 0, 0, 0.03125f, 0x00000000u},
      {0.5f, 0, 0, 0, 0x40800000u},
      {0.5f, 0.125f, 0, 0.03125f, 0x40800000u},
      {0.25f, 0.125f, 0.125f, 0.25f, 0x406b0000u},
      // The same-face and explicit-gradient controls retain magnitude rounding.
      {-0x1.3d50000000000p-8f, -0x1.2bb5a00000000p-3f, -0x1.531b800000000p-5f,
       -0x1.0aa0000000000p-3f, 0x40358000u},
      {0x1.09cb800000000p-5f, -0x1.93e0000000000p-5f, 0x1.4d64800000000p-5f, -0x1.4ced000000000p-5f,
       0x3fa50000u},
  };
  for (uint32_t opcode : {27u, 28u})
    for (const auto &witness : witnesses) {
      SCOPED_TRACE(testing::Message() << opcode << ", " << witness.dxu << ", " << witness.dyv);
      wave_->set_exec(15);
      wave_->debug_write_vgpr(12, 31, 0xdeadbeef);
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t offset = opcode == 28 ? 4 : 0;
        const float gradients[] = {witness.dxu, witness.dxv, witness.dyu, witness.dyv};
        for (uint32_t i = 0; i < offset; ++i)
          wave_->debug_write_vgpr(i, lane, std::bit_cast<uint32_t>(gradients[i] * 0.5f));
        const float u = 0.125f + (lane & 1) * witness.dxu + (lane >> 1) * witness.dyu;
        const float v = 0.25f + (lane & 1) * witness.dxv + (lane >> 1) * witness.dyv;
        wave_->debug_write_vgpr(offset, lane, std::bit_cast<uint32_t>(1.5f + u * 0.5f));
        wave_->debug_write_vgpr(offset + 1, lane, std::bit_cast<uint32_t>(1.5f + v * 0.5f));
        wave_->debug_write_vgpr(offset + 2, lane, 0);
      }
      ASSERT_NO_FATAL_FAILURE(sample(opcode, 3));
      for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t c = 0; c < 4; ++c)
          EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witness.expected);
      EXPECT_EQ(wave_->debug_read_vgpr(12, 31), 0xdeadbeefu);
    }

  // One quad spans +Y, +X and +Z. Both cards share the origin's footprint
  // across all lanes; unfolding onto lane 3's face instead gives mip 3.
  const float corner[4][3] = {{1.9375f, 1.96875f, 2.0f},
                              {1.90625f, 1.9375f, 2.0f},
                              {1.0625f, 1.03125f, 0.0f},
                              {1.9375f, 1.03125f, 4.0f}};
  wave_->set_exec(15);
  for (uint32_t lane = 0; lane < 4; ++lane)
    for (uint32_t c = 0; c < 3; ++c)
      wave_->debug_write_vgpr(c, lane, std::bit_cast<uint32_t>(corner[lane][c]));
  ASSERT_NO_FATAL_FAILURE(sample(27, 3));
  for (uint32_t lane = 0; lane < 4; ++lane)
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), 0x40390000u);
  // Captured halfway cases distinguish source-face rounding from magnitude
  // rounding and cover both signs of the unfolded coordinate transform.
  struct CornerWitness {
    uint32_t coordinates[4][3];
    uint32_t expected;
  };
  const CornerWitness corner_witnesses[] = {
      {{{0x3f859aefu, 0x3f82ceeau, 0x00000000u},
        {0x3ff9713du, 0x3ffab465u, 0x40000000u},
        {0x3ffa79eau, 0x3ffd0b48u, 0x40000000u},
        {0x3ffb9eacu, 0x3f856c1fu, 0x40800000u}},
       0x40350000u},
      {{{0x3ffd516du, 0x3f82e7cdu, 0x40400000u},
        {0x3fff6504u, 0x3fffc00du, 0x40800000u},
        {0x3fffec36u, 0x3f804df3u, 0x40400000u},
        {0x3fff33d9u, 0x3fff943eu, 0x40800000u}},
       0x3fa40000u},
      {{{0x3ffa6511u, 0x3f82ceeau, 0x3f800000u},
        {0x3f868ec3u, 0x3ffab465u, 0x40000000u},
        {0x3f858616u, 0x3ffd0b48u, 0x40000000u},
        {0x3f846154u, 0x3f856c1fu, 0x40800000u}},
       0x40358000u},
      {{{0x3f859aefu, 0x3ffd3116u, 0x00000000u},
        {0x3ff9713du, 0x3f854b9bu, 0x40400000u},
        {0x3ffa79eau, 0x3f82f4b8u, 0x40400000u},
        {0x3ffb9eacu, 0x3ffa93e1u, 0x40800000u}},
       0x40350000u},
      {{{0x3ffd516du, 0x3ffd1833u, 0x40400000u},
        {0x3f809afcu, 0x3fffc00du, 0x40a00000u},
        {0x3fffec36u, 0x3fffb20du, 0x40400000u},
        {0x3f80cc27u, 0x3fff943eu, 0x40a00000u}},
       0x3fa40000u},
      {{{0x3ffa6511u, 0x3ffd3116u, 0x00000000u},
        {0x3ff9713du, 0x3ffab465u, 0x40400000u},
        {0x3ffa79eau, 0x3ffd0b48u, 0x40400000u},
        {0x3f846154u, 0x3ffa93e1u, 0x40a00000u}},
       0x40350000u},
  };
  for (const auto &witness : corner_witnesses) {
    for (uint32_t lane = 0; lane < 4; ++lane)
      for (uint32_t c = 0; c < 3; ++c)
        wave_->debug_write_vgpr(c, lane, witness.coordinates[lane][c]);
    ASSERT_NO_FATAL_FAILURE(sample(27, 3));
    for (uint32_t lane = 0; lane < 4; ++lane)
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witness.expected);
  }
  // A non-power-of-two extent distinguishes the parallel-coordinate rule and
  // the reflected sum with 22 fractional bits before derivative conversion.
  setup_cube(127);
  const CornerWitness non_power_of_two[] = {
      {{{0x3ffad4f9u, 0x3ffe2842u, 0x40800000u},
        {0x3ff94b3bu, 0x3ff5b576u, 0x40800000u},
        {0x3f860d2au, 0x3ff95142u, 0x00000000u},
        {0x3ff7d038u, 0x3ffd6624u, 0x40800000u}},
       0x40710000u},
      {{{0x3ff98d25u, 0x3ff47332u, 0x40400000u},
        {0x3f897a5au, 0x3ff33712u, 0x40a00000u},
        {0x3ffa12a5u, 0x3ffe95a0u, 0x00000000u},
        {0x3f886537u, 0x3ffd4abau, 0x40a00000u}},
       0x40988000u},
      {{{0x3fff5614u, 0x3f8042ecu, 0x40400000u},
        {0x3f80c7d0u, 0x3fffebfau, 0x00000000u},
        {0x3ffedf04u, 0x3fff3680u, 0x40800000u},
        {0x3ffe93e6u, 0x3f80aeb5u, 0x40400000u}},
       0x3ef40000u},
      {{{0x3f80d7d5u, 0x3fff64bcu, 0x40400000u},
        {0x3f807792u, 0x3fff4c93u, 0x40400000u},
        {0x3f802f9bu, 0x3fffbd84u, 0x3f800000u},
        {0x3fffdf89u, 0x3fffbeddu, 0x40a00000u}},
       0x3d800000u},
  };
  for (const auto &witness : non_power_of_two) {
    for (uint32_t lane = 0; lane < 4; ++lane)
      for (uint32_t c = 0; c < 3; ++c)
        wave_->debug_write_vgpr(c, lane, witness.coordinates[lane][c]);
    ASSERT_NO_FATAL_FAILURE(sample(27, 3));
    for (uint32_t lane = 0; lane < 4; ++lane)
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, lane), witness.expected);
  }
}

TEST_P(GraphicsExportTest, Fp32FilteringMatchesPhysicalRoundingAndSpecialValues) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Witness {
    uint32_t pattern, query;
    std::array<uint32_t, 4> expected;
  };
  const Witness witnesses[] = {
      {0, 0, {0x3ee5747au, 0x3e0e5bc4u, 0x3e12bb23u, 0x3e379a0au}},
      {0, 1, {0x3ee570cbu, 0x3e0e5f92u, 0x3e12df73u, 0x3e37bad8u}},
      {0, 255, {0x3ee1c906u, 0x3e122661u, 0x3e36e6b6u, 0x3e58478eu}},
      {0, 256, {0x3ee56c9du, 0x3e0e8420u, 0x3e12c441u, 0x3e37df6au}},
      {0, 257, {0x3ee5686fu, 0x3e0e87cfu, 0x3e12e8cdu, 0x3e37fffbu}},
      {0, 32896, {0x3ebff983u, 0x3e1ca19eu, 0x3e3882c4u, 0x3e5b6214u}},
      {0, 65535, {0x3e383a88u, 0x3e1b5aa7u, 0x3e7bc5b7u, 0x3e609f98u}},
      {1, 0, {0xbee5747au, 0xbe0e5bc4u, 0xbe12bb23u, 0xbe379a0au}},
      {1, 1, {0xbee570cbu, 0xbe0e5f92u, 0xbe12df73u, 0xbe37bad8u}},
      {1, 255, {0xbee1c906u, 0xbe122661u, 0xbe36e6b6u, 0xbe58478eu}},
      {1, 256, {0xbee56c9du, 0xbe0e8420u, 0xbe12c441u, 0xbe37df6au}},
      {1, 257, {0xbee5686fu, 0xbe0e87cfu, 0xbe12e8cdu, 0xbe37fffbu}},
      {1, 32896, {0xbebff983u, 0xbe1ca19eu, 0xbe3882c4u, 0xbe5b6214u}},
      {1, 65535, {0xbe383a88u, 0xbe1b5aa7u, 0xbe7bc5b7u, 0xbe609f98u}},
      {2, 0, {0xc665747au, 0xc70e5bc4u, 0xc712bb23u, 0xc7379a0au}},
      {2, 1, {0xc6649d22u, 0xc70dd68bu, 0xc71233d9u, 0xc736eff6u}},
      {2, 255, {0xc46f3ad2u, 0xc51a7dbau, 0xc53f7fa7u, 0xc5630990u}},
      {2, 256, {0xc6648fe3u, 0xc70dce1fu, 0xc7122904u, 0xc736e36du}},
      {2, 257, {0xc663b962u, 0xc70d496au, 0xc711a240u, 0xc7363a03u}},
      {2, 32896, {0xc5749bf0u, 0xc6183ed3u, 0xc61ed772u, 0xc6462b92u}},
      {2, 65535, {0xc173fe02u, 0xc19ff7edu, 0xc1e1d1d3u, 0xc1e8c899u}},
      {3, 0, {0xffc00000u, 0x00000000u, 0x80000000u, 0xffc00000u}},
      {3, 1, {0xffc00000u, 0xbc000000u, 0x3b800000u, 0xffc00000u}},
      {3, 255, {0xffc00000u, 0xbfff0000u, 0x3f7f0000u, 0xffc00000u}},
      {3, 256, {0xffc00000u, 0x00000000u, 0xff800000u, 0xffc00000u}},
      {3, 257, {0xffc00000u, 0xbbff0000u, 0xffc00000u, 0xffc00000u}},
      {3, 32896, {0xffc00000u, 0xbf000000u, 0xffc00000u, 0xffc00000u}},
      {3, 65535, {0xffc00000u, 0xbbff0000u, 0xffc00000u, 0xffc00000u}},
  };
  for (uint32_t pattern = 0; pattern < 4; ++pattern) {
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    for (uint32_t t = 0; t < 4; ++t)
      for (uint32_t c = 0; c < 4; ++c) {
        uint32_t q = t * 131 + c * 7 + 12345;
        q ^= q << 13;
        q ^= q >> 17;
        q ^= q << 5;
        uint32_t bits = 0x3e000000 + (q & 0xffffff);
        if (pattern == 1)
          bits |= q & 0x80000000;
        if (pattern == 2)
          bits = q & 0xff7fffff;
        if (pattern == 3) {
          constexpr uint32_t special[]{0,          0x80000000, 1,          0x80000001,
                                       0x7fffff,   0x807fffff, 0x3f800000, 0xbf800000,
                                       0x7f800000, 0xff800000, 0x7f800001, 0xff800001,
                                       0x7fc00001, 0xffc00001, 0x40000000, 0xc0000000};
          bits = special[q & 15];
        }
        const auto address = gfx12 ? amdgpu::gfx12_image_address(0x100000, t % 2, t / 2, 16, 16, 0)
                                   : amdgpu::gfx11_image_address(0x100000, t % 2, t / 2, 16, 16, 0);
        ASSERT_TRUE(address);
        memory_.write32(*address + 4 * c, bits);
      }
    const std::array<uint32_t, 8> descriptor{
        0x1000, (63u << (gfx12 ? 17 : 20)) | (1u << 30), 1u << 14, (9u << 28) | 0xfac, 15, 0, 0, 0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(4, 0x92);
    wave_->debug_write_sgpr(5, 0);
    wave_->debug_write_sgpr(6, (1u << 20) | (1u << 22));
    wave_->debug_write_sgpr(7, 0);
    wave_->set_exec(1);
    for (const auto &witness : witnesses) {
      if (witness.pattern != pattern)
        continue;
      wave_->debug_write_vgpr(0, 0,
                              std::bit_cast<uint32_t>(0.25f + (witness.query & 255) / 512.0f));
      wave_->debug_write_vgpr(1, 0, std::bit_cast<uint32_t>(0.25f + (witness.query >> 8) / 512.0f));
      ASSERT_NO_FATAL_FAILURE(sample(31, 1));
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), witness.expected[c])
            << pattern << "," << witness.query << "," << c;
    }
  }
}

TEST_P(GraphicsExportTest, AnisotropicCountsAndSpacingMatchPhysicalReadbacks) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, 64, 64, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 64; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12
                               ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                               : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
      ASSERT_TRUE(address);
      const float channels[] = {float(std::max(int(x) - 32, 0)), float(std::max(32 - int(x), 0)),
                                float(std::max(int(y) - 32, 0)), float(std::max(32 - int(y), 0))};
      for (uint32_t c = 0; c < 4; ++c)
        memory_.write32(*address + 4 * c, std::bit_cast<uint32_t>(channels[c]));
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (63u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (63u << 14),
                                           (9u << 28) | 0xfac,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
  wave_->debug_write_sgpr(7, 0);
  struct Witness {
    uint32_t max_anisotropy, bias, threshold;
    std::array<uint32_t, 4> gradients;
    int32_t phase;
    std::array<uint32_t, 4> expected;
    uint32_t mip_filter = 0;
  };
  // Identical raw RGBA32F readbacks on physical GFX11 and GFX12. Rotated
  // footprints cross a count boundary and exercise every even count, both
  // reconstruction tables, and bias redistribution with threshold controls.
  constexpr Witness witnesses[] = {
      {16,
       0,
       0,
       {0xbe940e97u, 0xbf7a0389u, 0x3e36996cu, 0xbf304c5fu},
       -26,
       {0x3c600000u, 0x3c800000u, 0x3dfc0000u, 0x3e000000u}},
      {16,
       0,
       0,
       {0x3e129a71u, 0xbfa7d059u, 0x3e9ec2fau, 0xbff798deu},
       -20,
       {0x3d280000u, 0x3d340000u, 0x3e960000u, 0x3e970000u}},
      {16,
       0,
       0,
       {0xbfdd0965u, 0x402879d4u, 0xc01d1f9bu, 0x3fcafaecu},
       -14,
       {0x3ebcc100u, 0x3ebcc100u, 0x3ec16a00u, 0x3ec21480u}},
      {16,
       0,
       0,
       {0x3e2c99a1u, 0x3ed1cb53u, 0xbfbc8651u, 0x40d63183u},
       -8,
       {0x3e3b0000u, 0x3e3d0000u, 0x3f574000u, 0x3f578000u}},
      {16,
       0,
       0,
       {0xc090a0f2u, 0xc0e23c12u, 0x4039b9cdu, 0x407fe00au},
       -2,
       {0x3f2c5d80u, 0x3f2c5d80u, 0x3f8265e0u, 0x3f82a5e0u}},
      {16,
       0,
       0,
       {0x403aa0e4u, 0xc0cae6ceu, 0xc03b9eecu, 0x40f3647bu},
       4,
       {0x3f046920u, 0x3f043e80u, 0x3f9eeaf0u, 0x3f9ed5a0u}},
      {16,
       0,
       0,
       {0xc0f7ff44u, 0x40d9d8f5u, 0xc0fa09a6u, 0x409b77a6u},
       10,
       {0x3fb077c0u, 0x3fb04a10u, 0x3f85a290u, 0x3f857df0u}},
      {16,
       0,
       0,
       {0xc03e060bu, 0x3cf68e78u, 0xc19962d1u, 0xc03d5fbeu},
       16,
       {0x401a0000u, 0x401a0000u, 0x3eb9e000u, 0x3eb8e000u}},
      {4,
       0,
       0,
       {0xbfa52c6cu, 0x3fe6900au, 0xbf8beb8du, 0xc033bf44u},
       -31,
       {0x3d140000u, 0x3d200000u, 0x3ed48000u, 0x3ed68000u}},
      {16,
       16,
       0,
       {0xc0f7ff44u, 0x40d9d8f5u, 0xc0fa09a6u, 0x409b77a6u},
       1,
       {0x3fafa560u, 0x3fafa560u, 0x3f84f3a0u, 0x3f84f3a0u}},
      {16,
       16,
       0,
       {0xbfdd0965u, 0x402879d4u, 0xc01d1f9bu, 0x3fcafaecu},
       7,
       {0x3ebd8000u, 0x3ebc8000u, 0x3ec20000u, 0x3ec20000u}},
      {16,
       16,
       0,
       {0xc06206b2u, 0x40ad7d2bu, 0xbfe6d7a8u, 0x3f9c226fu},
       14,
       {0x3efaac80u, 0x3ef9ac80u, 0x3f31c4c0u, 0x3f3144c0u}},
      {16,
       16,
       0,
       {0x3feb0040u, 0xc0b85fcau, 0x41312b68u, 0xc10a47e0u},
       3,
       {0x3faf2000u, 0x3faf2000u, 0x3fa0e000u, 0x3fa0a000u}},
      {16,
       16,
       0,
       {0xc089f1dbu, 0x4146bc0bu, 0xc083611du, 0x40e70d25u},
       8,
       {0x3f38a7e0u, 0x3f3874a0u, 0x3fe35280u, 0x3fe35280u}},
      {16,
       16,
       0,
       {0xbf7c5ad0u, 0x40026b1eu, 0xc00cf53fu, 0x3efd8518u},
       3,
       {0x3e900000u, 0x3e900000u, 0x3e660000u, 0x3e640000u}},
      {16,
       63,
       0,
       {0xc089f1dbu, 0x4146bc0bu, 0xc083611du, 0x40e70d25u},
       -1,
       {0x3f394240u, 0x3f394240u, 0x3fe3f3e0u, 0x3fe433e0u}},
      {16,
       63,
       7,
       {0xc089f1dbu, 0x4146bc0bu, 0xc083611du, 0x40e70d25u},
       -1,
       {0x3f36c300u, 0x3f36c300u, 0x3fe0f4e0u, 0x3fe134e0u}},
      // Linear mip filtering retains finer span precision, including when
      // both mip taps select this single-level image.
      {2,
       0,
       0,
       {0xbf82e328u, 0x40d08320u, 0x3ebc90a6u, 0xc0568304u},
       -26,
       {0x3e080000u, 0x3e0a0000u, 0x3f680000u, 0x3f690000u},
       2},
      {4,
       0,
       0,
       {0x407ea1c2u, 0xc03d4ca7u, 0x40211620u, 0xbfc9eb88u},
       15,
       {0x3f170000u, 0x3f168000u, 0x3ed70000u, 0x3ed60000u},
       2},
      {8,
       0,
       0,
       {0xc0ca76f8u, 0xc12aa528u, 0xbf84d27du, 0xc0a4fa45u},
       15,
       {0x3f47a000u, 0x3f472000u, 0x3fbca000u, 0x3fbc6000u},
       2},
      {16,
       0,
       0,
       {0x3fd0c0bbu, 0xbed1ef98u, 0xc0301859u, 0x411b09a0u},
       -7,
       {0x3eb88000u, 0x3eb90000u, 0x3f9b9000u, 0x3f9bb000u},
       2},
      {4,
       16,
       0,
       {0x407ea1c2u, 0xc03d4ca7u, 0x40211620u, 0xbfc9eb88u},
       30,
       {0x3f174000u, 0x3f164000u, 0x3ed70000u, 0x3ed58000u},
       2},
      {16,
       16,
       0,
       {0x40c2c47cu, 0xbf2cb363u, 0x41830fbcu, 0x406e7efdu},
       8,
       {0x400c0000u, 0x400c0000u, 0x3ed48000u, 0x3ed40000u},
       2},
      {16,
       0,
       7,
       {0x3fd0c0bbu, 0xbed1ef98u, 0xc0301859u, 0x411b09a0u},
       -7,
       {0x3eb88000u, 0x3eb90000u, 0x3f9b9000u, 0x3f9bb000u},
       2},
  };
  wave_->set_exec(1);
  for (const auto &witness : witnesses) {
    wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22) | (witness.mip_filter << 26));
    wave_->debug_write_sgpr(4, 0x92 | (std::countr_zero(witness.max_anisotropy) << 9) |
                                   (witness.threshold << 16) | (witness.bias << 21) | (1u << 27));
    for (uint32_t i = 0; i < 4; ++i)
      wave_->debug_write_vgpr(
          i, 0, std::bit_cast<uint32_t>(std::bit_cast<float>(witness.gradients[i]) / 64));
    const float coordinate = (32.5f + witness.phase / 8192.0f) / 64;
    wave_->debug_write_vgpr(4, 0, std::bit_cast<uint32_t>(coordinate));
    wave_->debug_write_vgpr(5, 0, std::bit_cast<uint32_t>(coordinate));
    ASSERT_NO_FATAL_FAILURE(sample(28, 1));
    for (uint32_t c = 0; c < 4; ++c)
      EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), witness.expected[c])
          << witness.max_anisotropy << ',' << witness.bias << ',' << witness.phase << ',' << c;
  }
}

TEST_P(GraphicsExportTest, AnisotropicFloatingAccumulationMatchesPhysicalReadbacks) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  struct Witness {
    uint32_t format, max_anisotropy, bias;
    std::array<uint32_t, 4> gradients;
    int32_t phase;
    std::array<uint32_t, 4> expected;
  };
  // Raw physical GFX11/GFX12 results for a seeded texture. sRGB and FP16
  // expose intermediate accumulation; FP32 cancellation also distinguishes
  // operand alignment and retention of the accumulator exponent.
  constexpr Witness witnesses[] = {
      {66,
       16,
       0,
       {0xc129dff5u, 0xc01193ceu, 0xc0d91b8fu, 0xc0884878u},
       -29,
       {0x3e877469u, 0x3e679331u, 0x3e3cad84u, 0x3eac1192u}},
      {66,
       16,
       16,
       {0xc129dff5u, 0xc01193ceu, 0xc0d91b8fu, 0xc0884878u},
       -29,
       {0x3e877469u, 0x3e679331u, 0x3e3cad84u, 0x3eac1192u}},
      {57,
       2,
       0,
       {0xc12fda97u, 0x41a04872u, 0xc1221509u, 0x415ec7cau},
       1,
       {0xc2fd81d6u, 0x40e16010u, 0x41b1f62cu, 0xc193d850u}},
      {57,
       4,
       0,
       {0xc03e060bu, 0x3cf68e78u, 0xc19962d1u, 0xc03d5fbeu},
       -16,
       {0xc39c9d75u, 0x409e393eu, 0xc2c4018cu, 0x42e95d51u}},
      {57,
       16,
       16,
       {0xc03e060bu, 0x3cf68e78u, 0xc19962d1u, 0xc03d5fbeu},
       -29,
       {0xc1eb252cu, 0x42966696u, 0xc2c9e0beu, 0x41e5f944u}},
      {63,
       2,
       0,
       {0xbf6dd938u, 0xbe93684bu, 0xbeae4847u, 0xbe863b6bu},
       -29,
       {0xc267165du, 0xc25ed558u, 0xc6c43c87u, 0xc1edd99cu}},
      {63,
       2,
       0,
       {0xc0c6768eu, 0xc085be2cu, 0x41e515d2u, 0x40c52df9u},
       9,
       {0xc561eb8bu, 0x3c42e800u, 0xc38770b3u, 0xc5383b8au}},
      {63,
       4,
       0,
       {0x3e129a71u, 0xbfa7d059u, 0x3e9ec2fau, 0xbff798deu},
       9,
       {0xc2f836e8u, 0x43f87c95u, 0xc6259cecu, 0xc532fdb2u}},
      {63,
       4,
       0,
       {0x3f533b6fu, 0x3ec0655du, 0xbf806c3bu, 0xbed562d0u},
       20,
       {0x3bd67940u, 0xc3c113b0u, 0xc637d948u, 0xc44d624fu}},
      {63,
       16,
       0,
       {0x3e129a71u, 0xbfa7d059u, 0x3e9ec2fau, 0xbff798deu},
       9,
       {0xc2f836e8u, 0x43f87c95u, 0xc6259cecu, 0xc532fdb2u}},
      {63,
       16,
       0,
       {0xc112e8d2u, 0xc12b9ff7u, 0x40a16424u, 0x40ae589cu},
       3,
       {0xc3faa53eu, 0xc201b8f1u, 0x3ef1d500u, 0x42afe091u}},
      {63,
       16,
       16,
       {0x3e129a71u, 0xbfa7d059u, 0x3e9ec2fau, 0xbff798deu},
       9,
       {0xc2f836e8u, 0x43f87c95u, 0xc6259cecu, 0xc532fdb2u}},
      {63,
       16,
       16,
       {0xc112e8d2u, 0xc12b9ff7u, 0x40a16424u, 0x40ae589cu},
       3,
       {0xc3faa53eu, 0xc201b8f1u, 0x3ef1d500u, 0x42afe091u}},
  };
  for (uint32_t format : {66u, 57u, 63u}) {
    cu_->l1_vector().invalidate_all();
    cache_.invalidate_all();
    const uint32_t bytes = format == 66 ? 4 : format == 57 ? 8 : 16;
    const auto layout = amdgpu::image_mip_layout(gfx12, 0, bytes, 64, 64, 1, 0);
    ASSERT_TRUE(layout);
    uint32_t state = 0x7941332d;
    const auto next = [&]() {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      return state;
    };
    for (uint32_t y = 0; y < 64; ++y)
      for (uint32_t x = 0; x < 64; ++x) {
        const auto address =
            gfx12 ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, bytes, 0)
                  : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, bytes, 0);
        ASSERT_TRUE(address);
        if (format == 66) {
          memory_.write32(*address, next());
          continue;
        }
        for (uint32_t c = 0; c < 4; ++c) {
          const uint32_t bits = next();
          if (format == 57)
            memory_.write16(*address + c * 2, (bits & 0x83ff) | (((bits >> 10) % 24 + 3) << 10));
          else
            memory_.write32(*address + c * 4,
                            (bits & 0x807fffff) | (((bits >> 23) % 32 + 111) << 23));
        }
      }
    const std::array<uint32_t, 8> descriptor{0x1000,
                                             (format << (gfx12 ? 17 : 20)) | (3u << 30),
                                             15 | (63u << 14),
                                             (9u << 28) | 0xfac,
                                             0,
                                             4u << 20,
                                             0,
                                             0};
    for (uint32_t i = 0; i < descriptor.size(); ++i)
      wave_->debug_write_sgpr(8 + i, descriptor[i]);
    wave_->debug_write_sgpr(5, 0);
    wave_->debug_write_sgpr(6, (3u << 20) | (3u << 22));
    wave_->debug_write_sgpr(7, 0);
    wave_->set_exec(1);
    for (const auto &witness : witnesses) {
      if (witness.format != format)
        continue;
      wave_->debug_write_sgpr(4, 0x92 | (std::countr_zero(witness.max_anisotropy) << 9) |
                                     (witness.bias << 21) | (1u << 27));
      for (uint32_t i = 0; i < 4; ++i)
        wave_->debug_write_vgpr(
            i, 0, std::bit_cast<uint32_t>(std::bit_cast<float>(witness.gradients[i]) / 64));
      const float coordinate = (32.5f + witness.phase / 8192.0f) / 64;
      wave_->debug_write_vgpr(4, 0, std::bit_cast<uint32_t>(coordinate));
      wave_->debug_write_vgpr(5, 0, std::bit_cast<uint32_t>(coordinate));
      ASSERT_NO_FATAL_FAILURE(sample(28, 1));
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0), witness.expected[c])
            << format << ',' << witness.max_anisotropy << ',' << witness.bias << ','
            << witness.phase << ',' << c;
    }
  }
}

TEST_P(GraphicsExportTest, AnisotropicFloatingSpecialValuesMatchPhysicalReadbacks) {
  const bool gfx12 = GetParam() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t pairs[][2] = {
      {0x00000000u, 0x00000000u}, {0x80000000u, 0x80000000u}, {0x00000000u, 0x80000000u},
      {0x80000000u, 0x00000000u}, {0x3f800000u, 0xbf800000u}, {0x7f800000u, 0x3f800000u},
      {0xff800000u, 0x3f800000u}, {0x7f800000u, 0x7f800000u}, {0xff800000u, 0xff800000u},
      {0x7f800000u, 0xff800000u}, {0x7fc12345u, 0x3f800000u}, {0xffc12345u, 0x3f800000u},
      {0x7f812345u, 0x00000000u}, {0x00000001u, 0x00000000u}, {0x80000001u, 0x00000000u},
      {0x00800000u, 0x80800000u}, {0x3f800001u, 0xbf800000u}, {0x3f800000u, 0xbf800001u},
      {0x3f800000u, 0x33800000u}, {0xbf800000u, 0x33800000u}, {0x40000000u, 0xbfffffffu},
      {0x00800000u, 0x00000000u}, {0x80800000u, 0x00000000u}, {0x7f7fffffu, 0xff7fffffu},
  };
  // Linear and nearest anisotropic filters, captured on both physical targets.
  // Include signed zeros, infinities, NaNs, cancellation and output underflow.
  constexpr uint32_t expected[2][std::size(pairs)] = {
      {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x7f800000u,
       0xff800000u, 0x7f800000u, 0xff800000u, 0xffc00000u, 0xffc00000u, 0xffc00000u,
       0xffc00000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x33000000u, 0xb3000000u,
       0x3e800000u, 0xbe7fffffu, 0x33000000u, 0x00000000u, 0x80000000u, 0x00000000u},
      {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0xbf000000u, 0x3f000000u,
       0x3f000000u, 0x7f800000u, 0xff800000u, 0xff800000u, 0x3f000000u, 0x3f000000u,
       0x00000000u, 0x00000000u, 0x00000000u, 0x80000000u, 0xbf000000u, 0xbf000001u,
       0x33000000u, 0x33000000u, 0xbf7fffffu, 0x00000000u, 0x00000000u, 0xfeffffffu},
  };
  const auto layout = amdgpu::image_mip_layout(gfx12, 0, 16, 64, 64, 1, 0);
  ASSERT_TRUE(layout);
  for (uint32_t y = 0; y < 64; ++y)
    for (uint32_t x = 0; x < 64; ++x) {
      const auto address = gfx12
                               ? amdgpu::gfx12_image_address(0x100000, x, y, layout->pitch, 16, 0)
                               : amdgpu::gfx11_image_address(0x100000, x, y, layout->pitch, 16, 0);
      ASSERT_TRUE(address);
      for (uint32_t c = 0; c < 4; ++c) {
        const auto &pair = pairs[(y + c) % std::size(pairs)];
        const uint32_t value = x < 32   ? pair[0]
                               : x > 32 ? pair[1]
                                        : (pair[0] & pair[1] & 0x80000000u);
        memory_.write32(*address + c * 4, value);
      }
    }
  const std::array<uint32_t, 8> descriptor{0x1000,
                                           (63u << (gfx12 ? 17 : 20)) | (3u << 30),
                                           15 | (63u << 14),
                                           (9u << 28) | 0xfac,
                                           0,
                                           4u << 20,
                                           0,
                                           0};
  for (uint32_t i = 0; i < descriptor.size(); ++i)
    wave_->debug_write_sgpr(8 + i, descriptor[i]);
  wave_->debug_write_sgpr(4, 0x92 | (1u << 9) | (1u << 27));
  wave_->debug_write_sgpr(5, 0);
  wave_->debug_write_sgpr(7, 0);
  wave_->set_exec(1);
  const float gradients[] = {2.0f / 64, 0, 0, 1.0f / 64};
  for (uint32_t i = 0; i < 4; ++i)
    wave_->debug_write_vgpr(i, 0, std::bit_cast<uint32_t>(gradients[i]));
  wave_->debug_write_vgpr(4, 0, std::bit_cast<uint32_t>(32.5f / 64));
  for (uint32_t nearest : {0u, 1u}) {
    const uint32_t filter = nearest ? 2 : 3;
    wave_->debug_write_sgpr(6, (filter << 20) | (filter << 22));
    for (uint32_t row = 0; row < std::size(pairs); ++row) {
      wave_->debug_write_vgpr(5, 0, std::bit_cast<uint32_t>((row + 0.5f) / 64));
      ASSERT_NO_FATAL_FAILURE(sample(28, 1));
      for (uint32_t c = 0; c < 4; ++c)
        EXPECT_EQ(wave_->debug_read_vgpr(12 + c, 0),
                  expected[nearest][(row + c) % std::size(pairs)])
            << nearest << ',' << row << ',' << c;
    }
  }
}
