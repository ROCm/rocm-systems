// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "compute_unit_test_util.h"
#include "gfx11_mimg_test_common.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/mimg.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
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
#include "rocjitsu/isa/arch/amdgpu/shared/image_access.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_address.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_footprint.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_resource.h"
#include "rocjitsu/isa/arch/amdgpu/shared/image_transfer.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/image_addressing.h"
#include "rocjitsu/vm/amdgpu/instruction_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/graphics/image_address.h"
#include "rocjitsu/vm/graphics/image_metadata.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/data_types.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <utility>

namespace {
using namespace rocjitsu;

TEST(ImageMetadataTest, RetryAReadOrPartialClearExpansionBeforePublishingTheDccKey) {
  using amdgpu::VmAccessOutcome;
  class Backing {
  public:
    mutable uint8_t key = 2; // Fast clear to all ones.
    mutable uint32_t operation = 0;
    uint32_t fail_once = 0;
    mutable std::array<uint8_t, 65536> pixels{};
    VmAccessOutcome read(uint64_t, std::span<std::byte> dst) const {
      if (++operation == fail_once)
        return VmAccessOutcome::Unavailable;
      dst.front() = std::byte{key};
      return VmAccessOutcome::Complete;
    }
    VmAccessOutcome write(uint64_t address, std::span<const std::byte> src) const {
      if (++operation == fail_once)
        return VmAccessOutcome::Unavailable;
      if (address == pixels.size())
        key = std::to_integer<uint8_t>(src.front());
      else
        std::memcpy(pixels.data() + address, src.data(), src.size());
      return VmAccessOutcome::Complete;
    }
  };
  for (const unsigned fail_once : {1u, 4u}) {
    Backing memory;
    memory.fail_once = fail_once;
    const auto materialize = [&] {
      return amdgpu::materialize_gfx11_dcc_at_address(memory, 0, memory.pixels.size(), 0, 0, 8, 8,
                                                      4, 27);
    };
    const auto first = materialize();
    ASSERT_TRUE(first.succeeded());
    EXPECT_EQ(first.value(), VmAccessOutcome::Unavailable);
    EXPECT_EQ(memory.key, 2);
    const auto second = materialize();
    ASSERT_TRUE(second.succeeded());
    EXPECT_EQ(second.value(), VmAccessOutcome::Complete);
    EXPECT_EQ(memory.key, 0xff);
    for (uint32_t y = 0; y < 8; ++y)
      for (uint32_t x = 0; x < 8; ++x) {
        const auto address = amdgpu::gfx11_image_address(0, x, y, 8, 4, 27);
        ASSERT_TRUE(address);
        for (uint32_t byte = 0; byte < 4; ++byte)
          EXPECT_EQ(memory.pixels[*address + byte], 0xff);
      }
  }
}

TEST(ImageFilterPolicyTest, BorderAndDepthComparisonsUseEncodedFunctionOrder) {
  const std::array<bool, 8> less{false, true, false, true, false, true, false, true};
  const std::array<bool, 8> equal{false, false, true, true, false, false, true, true};
  for (uint32_t function = 0; function < less.size(); ++function) {
    EXPECT_EQ(amdgpu::image_depth_comparison(function, 1, 2), less[function]);
    EXPECT_EQ(amdgpu::image_depth_comparison(function, 2, 2), equal[function]);
  }
  for (uint32_t selector = 0; selector < 8; ++selector) {
    EXPECT_EQ(amdgpu::image_border_component_is_one(0, selector), selector == 1);
    EXPECT_EQ(amdgpu::image_border_component_is_one(1, selector), selector == 1 || selector == 7);
    EXPECT_EQ(amdgpu::image_border_component_is_one(2, selector), selector == 1 || selector >= 4);
  }
}

constexpr uint32_t kIdentity = 4 | (5 << 3) | (6 << 6) | (7 << 9);
constexpr uint32_t kXyzOne = 4 | (5 << 3) | (6 << 6) | (1 << 9);
constexpr uint32_t kZyxOne = 6 | (5 << 3) | (4 << 6) | (1 << 9);
constexpr uint32_t kXOne = 4 | (1 << 9); // X, 0, 0, 1 (an R-only format).
constexpr uint64_t kBase = 0x10000;
constexpr uint32_t kDfmt32 = 4;
constexpr uint32_t kDfmt8888 = 10;
constexpr uint32_t kNfmtUint = 4;
constexpr uint32_t kNfmtUnorm = 0;
constexpr uint32_t kR32Uint = 20;
constexpr uint32_t kRgba32Uint = 61;
constexpr uint32_t kGfx11Rgba8Uint = 46;

struct Image {
  uint32_t data_format = kDfmt32;
  uint32_t num_format = kNfmtUint;
  uint32_t width = 1, height = 1, depth = 1, pitch = 64;
  uint32_t type = amdgpu::kImage2d;
  uint32_t selectors = kXOne;
  uint32_t swizzle_mode = 0;
  uint32_t mip_levels = 1;
  uint32_t base_level = 0;
  uint32_t last_level = 0;
  uint32_t rdna_format = kR32Uint;
  uint32_t min_lod = 0;
  uint32_t perf_mod = 0;
  bool corner_samples = false;
};

std::array<uint32_t, 8> rdna_descriptor(const Image &image, rj_code_arch_t arch);

std::array<uint32_t, 8> descriptor(const Image &image) {
  std::array<uint32_t, 8> r{};
  const auto set = [&](uint32_t bit, uint32_t width, uint64_t value) {
    for (uint32_t i = 0; i < width; ++i)
      r[(bit + i) / 32] |= static_cast<uint32_t>((value >> i) & 1u) << ((bit + i) % 32);
  };
  set(0, 40, kBase >> 8);
  set(40, 12, image.min_lod);
  set(92, 3, image.perf_mod);
  set(52, 6, image.data_format);
  set(58, 4, image.num_format);
  set(64, 14, image.width - 1);
  set(78, 14, image.height - 1);
  set(96, 12, image.selectors);
  set(108, 4, image.base_level);
  set(112, 4, image.last_level);
  set(188, 4, image.mip_levels - 1);
  set(116, 5, image.swizzle_mode);
  set(124, 4, image.type);
  set(128, 13, image.depth - 1);
  set(141, 16, image.pitch - 1);
  return r;
}

struct RdnaImageDescriptor {
  uint32_t type = amdgpu::kImage2d;
  uint32_t format = kR32Uint;
  uint32_t selectors = kXOne;
  uint32_t width = 3;
  uint32_t height = 2;
  uint32_t pitch = 64;
  uint32_t depth = 4;
  bool gfx12_bit_157 = false;
};

std::array<uint32_t, 8> rdna_image_descriptor(rj_code_arch_t arch,
                                              const RdnaImageDescriptor &image = {}) {
  std::array<uint32_t, 8> words{};
  const auto set = [&](uint32_t bit, uint32_t width, uint64_t value) {
    for (uint32_t i = 0; i < width; ++i)
      words[(bit + i) / 32] |= static_cast<uint32_t>((value >> i) & 1u) << ((bit + i) % 32);
  };
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  set(0, 40, kBase >> 8);
  set(gfx12 ? 49 : 52, 8, image.format);
  set(62, gfx12 ? 16 : 14, image.width - 1);
  set(78, gfx12 ? 16 : 14, image.height - 1);
  set(96, 12, image.selectors);
  set(124, 4, image.type);
  if (amdgpu::image_resource_encodes_pitch(image.type))
    set(128, gfx12 ? 16 : 14, image.pitch - 1);
  else
    set(128, gfx12 ? 14 : 13, image.depth - 1);
  if (gfx12 && image.gfx12_bit_157)
    set(157, 1, 1);
  return words;
}

class ImageAccessFixture : public testing::Test {
protected:
  amdgpu::GpuMemory memory_{"image_memory"};
  amdgpu::L2Cache l2_{"image_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  amdgpu::Wavefront *wf_ = nullptr;
  uint32_t sb_ = 0, vb_ = 0;

  virtual rj_code_arch_t arch() const = 0;
  virtual std::optional<uint32_t> image_gb_addr_config() const { return std::nullopt; }
  void SetUp() override {
    l2_.set_backing_memory(&memory_);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch();
    cfg.image_gb_addr_config = image_gb_addr_config();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 32;
    cfg.lds_size_kb = 64;
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(cu_, "image_cu", cfg, &memory_, &l2_));
    wf_ = cu_->dispatch_wf(0, 0, 106, 32);
    ASSERT_NE(wf_, nullptr);
    sb_ = wf_->sgpr_alloc().base;
    vb_ = wf_->vgpr_alloc().base;
    wf_->set_exec(5); // Lane 1 must not read, store or receive a load result.
  }
  void TearDown() override {
    if (wf_)
      wf_->halt();
  }

  void write_coordinates(uint32_t lane, std::array<uint32_t, 3> coords) {
    for (uint32_t i = 0; i < coords.size(); ++i)
      cu_->write_vgpr(vb_ + i, lane, coords[i]);
  }
  void check_zero_dmask_atomics() {
    wf_->set_exec(1);
    const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
    auto decoder = Decoder::create(arch());
    const auto resource = rdna_image_descriptor(arch(), {.format = kR32Uint});
    for (unsigned i = 0; i < resource.size(); ++i)
      cu_->write_sgpr(sb_ + i, resource[i]);
    write_coordinates(0, {0, 0, 0});
    memory_.write32(kBase, 13);
    for (bool compare : {false, true}) {
      const auto gfx12_op =
          compare ? rdna4::kImageAtomicCmpswapVimage : rdna4::kImageAtomicAddUintVimage;
      const auto gfx11_op =
          compare ? rdna3_5::kImageAtomicCmpswapMimg : rdna3_5::kImageAtomicAddMimg;
      for (bool returning : {false, true}) {
        for (bool tfe : {false, true}) {
          for (bool d16 : {false, true}) {
            SCOPED_TRACE(testing::Message() << "cmpswap=" << compare << " returning=" << returning
                                            << " tfe=" << tfe << " d16=" << d16);
            auto decoded =
                gfx12 ? decoder->decode_window(rdna4::build_vimage(gfx12_op, {.dim = 1,
                                                                              .d16 = d16,
                                                                              .dmask = 0,
                                                                              .vdata = 8,
                                                                              .th = returning,
                                                                              .tfe = tfe,
                                                                              .vaddr1 = 1}))
                      : decoder->decode_window(rdna3_5::build_mimg(gfx11_op, {.dim = 1,
                                                                              .unorm = 1,
                                                                              .dmask = 0,
                                                                              .glc = returning,
                                                                              .d16 = d16,
                                                                              .vdata = 8,
                                                                              .tfe = tfe}));
            ASSERT_TRUE(decoded.succeeded());
            auto inst = std::move(decoded).value();
            const InstDefUse def_use(*inst, wf_->wf_size());
            for (uint16_t reg : {8, 9}) {
              EXPECT_FALSE(def_use.defs.contains({RegClass::VGPR, reg, 1}));
              EXPECT_FALSE(def_use.uses.contains({RegClass::VGPR, reg, 1}));
            }
            EXPECT_FALSE(inst->flags() & (CONDITIONAL_LAST_VGPR_DEF | RESERVES_UNWRITTEN_VGPR));
            if (!gfx12) {
              const auto assembly = inst->disassemble();
              EXPECT_NE(assembly.find(tfe || compare ? " v[8:9]," : " v8,"), std::string::npos)
                  << assembly;
            }
            cu_->write_vgpr(vb_ + 8, 0, 21);
            cu_->write_vgpr(vb_ + 9, 0, 13);
            wf_->clear_instruction_execution_error();
            ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
            EXPECT_EQ(wf_->instruction_execution_error(),
                      gfx12 && d16 ? amdgpu::InstructionExecutionError::UnimplementedInstruction
                                   : amdgpu::InstructionExecutionError::UnsupportedOperandValue);
            EXPECT_EQ(inst->data(), nullptr);
            EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 21u);
            EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 13u);
            EXPECT_EQ(memory_.read32(kBase), 13u);
          }
        }
      }
    }
  }

  void check_out_of_bounds_atomic_payload() {
    class PayloadReads final : public ExecutionPlugin {
    public:
      PayloadReads() : ExecutionPlugin("out_of_bounds_atomic_payload") {}
      void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                                 uint8_t) override {
        if (reg == wave->vgpr_alloc().base + 8)
          read_lanes |= lanes;
      }
      uint64_t read_lanes = 0;
    };
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observer = std::make_unique<PayloadReads>();
    auto *reads = observer.get();
    ASSERT_TRUE(group->add(std::move(observer)));
    cu_->set_plugin_group(group);

    const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
    const auto descriptor = rdna_image_descriptor(arch());
    for (unsigned i = 0; i < descriptor.size(); ++i)
      cu_->write_sgpr(sb_ + i, descriptor[i]);
    wf_->set_exec(1);
    write_coordinates(0, {100, 0, 0}); // Outside this 3 x 2 image.
    for (bool returns : {false, true}) {
      SCOPED_TRACE(returns);
      wf_->pc = returns ? 0x4040 : 0x4000;
      const auto install = [&](const auto &words) {
        memory_.load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), wf_->pc);
      };
      if (gfx12)
        install(rdna4::build_vimage(
            rdna4::kImageAtomicAddUintVimage,
            {.dim = 1, .dmask = 1, .vdata = 8, .th = uint8_t(returns), .vaddr1 = 1}));
      else
        install(
            rdna3_5::build_mimg(rdna3_5::kImageAtomicAddMimg,
                                {.dim = 1, .unorm = 1, .dmask = 1, .glc = returns, .vdata = 8}));
      auto &state = wf_->ensure_memory_wait_scoreboard();
      state.clear();
      const auto counter = amdgpu::WaitCounterKind::Ds;
      state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, 8, 1}, counter, 0xf});
      cu_->write_vgpr(vb_ + 8, 0, 41);
      reads->read_lanes = 0;
      const auto before = cu_->memory_wait_diagnostic_count();
      cu_->step();
      ASSERT_FALSE(wf_->instruction_execution_failed());
      EXPECT_EQ(reads->read_lanes, gfx12 ? 0u : 1u);
      // An out-of-bounds returning atomic still overwrites VDATA with zero.
      EXPECT_EQ(cu_->memory_wait_diagnostic_count() != before, !gfx12 || returns);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), returns ? 0u : 41u);
    }
  }
  void check_atomic_prt_access(bool compare_swap, uint32_t base_level) {
    SCOPED_TRACE(testing::Message() << "base_level=" << base_level);
    class Pages final : public amdgpu::AddressSpaceTranslator {
    public:
      bool mapped = true, sparse = false;
      amdgpu::IdentityAddressSpaceTranslator identity;
      amdgpu::VmTranslationResult translate(uint64_t address, size_t width,
                                            amdgpu::VmAccessKind kind) const override {
        return !mapped && (address >> 12) != (kBase >> 12)
                   ? amdgpu::VmTranslationResult{}
                   : identity.translate(address, width, kind);
      }
      std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
        static const auto hole = std::make_shared<const amdgpu::PrtRanges>(
            amdgpu::PrtRanges{{kBase + 4096, kBase + 8192}});
        return sparse ? hole : nullptr;
      }
    };
    amdgpu::GpuVm vm;
    auto pages = std::make_shared<Pages>();
    unsigned faults = 0;
    auto handle = vm.register_address_space(
        1, pages, std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_),
        [&](uint64_t, amdgpu::VmAccessKind) { ++faults; });
    ASSERT_TRUE(handle);
    cu_->set_gpu_vm(&vm);
    wf_->set_address_space(handle);
    const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
    // The last mip in an RDNA linear chain starts at kBase. Use the native pitch
    // for a GFX11 mip chain and an explicit full row for a single-level image.
    const uint32_t width = 2048u << base_level;
    const auto resource = rdna_descriptor({.width = width,
                                           .pitch = gfx12 || !base_level ? width : 1u,
                                           .mip_levels = base_level + 1,
                                           .base_level = base_level,
                                           .last_level = base_level},
                                          arch());
    for (unsigned i = 0; i < resource.size(); ++i)
      cu_->write_sgpr(sb_ + i, resource[i]);
    wf_->set_exec(3);
    write_coordinates(0, {0, 0, 0});
    write_coordinates(1, {1024, 0, 0});
    auto decoder = Decoder::create(arch());
    const uint16_t gfx12_opcode =
        compare_swap ? rdna4::kImageAtomicCmpswapVimage : rdna4::kImageAtomicAddUintVimage;
    const uint16_t gfx11_opcode =
        compare_swap ? rdna3_5::kImageAtomicCmpswapMimg : rdna3_5::kImageAtomicAddMimg;
    const uint8_t dmask = compare_swap ? 3 : 1;
    for (bool prt : {false, true}) {
      cu_->write_sgpr(sb_ + 5, uint32_t(prt) << (gfx12 ? 5 : 8));
      for (bool returns : {false, true}) {
        for (bool tfe : {false, true}) {
          for (unsigned mapping = 0; mapping < 4; ++mapping) {
            pages->mapped = (mapping & 1u) == 0;
            pages->sparse = mapping >= 2;
            SCOPED_TRACE(testing::Message() << "PRT=" << prt << " return=" << returns
                                            << " TFE=" << tfe << " mapping=" << mapping);
            faults = 0;
            for (unsigned lane = 0; lane < 2; ++lane) {
              cu_->write_vgpr(vb_ + 8, lane, 5);
              cu_->write_vgpr(vb_ + 9, lane, lane == 0 ? 17 : 31);
              cu_->write_vgpr(vb_ + 10, lane, 0xcafef00d);
            }
            memory_.write32(kBase, 17);
            memory_.write32(kBase + 4096, 31);
            auto inst = gfx12 ? decoder->decode_window(
                                    rdna4::build_vimage(gfx12_opcode, {.dim = 1,
                                                                       .dmask = dmask,
                                                                       .vdata = 8,
                                                                       .th = uint8_t(returns),
                                                                       .tfe = uint8_t(tfe),
                                                                       .vaddr1 = 1}))
                              : decoder->decode_window(
                                    rdna3_5::build_mimg(gfx11_opcode, {.dim = 1,
                                                                       .unorm = 1,
                                                                       .dmask = dmask,
                                                                       .glc = uint8_t(returns),
                                                                       .vdata = 8,
                                                                       .tfe = uint8_t(tfe)}));
            ASSERT_TRUE(inst.succeeded());
            ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
            amdgpu::GlobalMemPipeline pipe(&cu_->l1_vector(), &l2_);
            const bool fault = !pages->sparse && !pages->mapped;
            const bool missed = pages->sparse;
            EXPECT_EQ(pipe.issue(inst.value().release(), *wf_),
                      fault ? amdgpu::VmAccessOutcome::Faulted : amdgpu::VmAccessOutcome::Complete);
            EXPECT_EQ(faults, unsigned(fault));
            cu_->l1_vector().flush_all();
            l2_.flush_all();
            EXPECT_EQ(memory_.read32(kBase + 4096), fault || missed ? 31u
                                                    : compare_swap  ? 5u
                                                                    : 36u);
            const uint32_t failure_status = (base_level << 16) | 1u;
            EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 1), missed && returns && tfe ? failure_status : 31u);
            EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 0), 0xcafef00du);
            EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 1), 0xcafef00du);
            if (!fault) {
              EXPECT_EQ(memory_.read32(kBase), compare_swap ? 5u : 22u);
              EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), returns ? 17u : 5u);
              EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), returns ? missed ? 0u : 31u : 5u);
              EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 17u);
            }
          }
        }
      }
    }
    wf_->set_address_space({});
    cu_->set_gpu_vm(nullptr);
  }

  void fill_destination(uint32_t value) {
    for (uint32_t lane = 0; lane < 3; ++lane)
      for (uint32_t reg = 0; reg < 4; ++reg)
        cu_->write_vgpr(vb_ + 8 + reg, lane, value);
  }
};

class Gfx11BvhTest : public ImageAccessFixture, public testing::WithParamInterface<rj_code_arch_t> {
protected:
  enum RayFlag : uint32_t {
    kForceOpaque = 1u << 0,
    kForceNonOpaque = 1u << 1,
    kDoubleSided = 1u << 2,
    kFlip = 1u << 3,
    kCullBack = 1u << 4,
    kCullFront = 1u << 5,
    kCullOpaque = 1u << 6,
    kCullNonOpaque = 1u << 7,
    kSkipTriangles = 1u << 8,
    kSkipProcedural = 1u << 9,
  };
  static constexpr unsigned kTriangleVertices[4][3] = {{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}};
  static constexpr uint32_t kBvhTypeField = amdgpu::kGfx11BvhResourceType << 28;
  rj_code_arch_t arch() const override { return GetParam(); }

  void write_box16_bounds(uint64_t address, const std::array<float, 6> &bounds) {
    // Box16 packs min.xyz followed by max.xyz into three consecutive DWORDs.
    for (unsigned word = 0; word < 3; ++word) {
      const uint32_t low_half = util::f32_to_f16(bounds[word * 2]);
      const uint32_t high_half = util::f32_to_f16(bounds[word * 2 + 1]);
      memory_.write32(address + word * 4, low_half | (high_half << 16));
    }
  }

  void expect_source_wait(const Instruction &inst, unsigned lane, unsigned source, bool read) {
    SCOPED_TRACE(testing::Message() << "lane=" << lane << " source=v" << source);
    auto &wait = wf_->ensure_memory_wait_scoreboard();
    unsigned warnings = 0;
    wait.clear();
    wait.bind(0x200, &warnings, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    const auto counter = amdgpu::WaitCounterKind::Ds;
    wait.add({wait.issue(counter),
              0x100,
              uint64_t{1} << lane,
              {RegClass::VGPR, uint16_t(source), 1},
              counter,
              0xf});
    wait.check_instruction(inst, *wf_);
    EXPECT_EQ(warnings, unsigned(read));
    wait.clear();
  }

  void check_triangle(const std::array<uint32_t, 9> &vertices,
                      const std::array<uint32_t, 3> &origin,
                      const std::array<uint32_t, 3> &direction, float extent, bool barycentrics,
                      const std::array<uint32_t, 4> &expected) {
    auto decoder = Decoder::create(arch());
    wf_->set_exec(1);
    cu_->write_sgpr(sb_, kBase >> 8);
    // Preserve the W7900 capture's TYPE field independently of the decoder.
    cu_->write_sgpr(sb_ + 3, (8u << 28) | uint32_t(barycentrics) << 24);
    for (unsigned i = 0; i < vertices.size(); ++i)
      memory_.write32(kBase + i * 4, vertices[i]);
    memory_.write32(kBase + 60, 9);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    cu_->write_vgpr(vb_, 0, 0);
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(extent));
    for (unsigned axis = 0; axis < 3; ++axis) {
      cu_->write_vgpr(vb_ + 2 + axis, 0, origin[axis]);
      cu_->write_vgpr(vb_ + 5 + axis, 0, direction[axis]);
      cu_->write_vgpr(vb_ + 8 + axis, 0,
                      std::bit_cast<uint32_t>(1.f / std::bit_cast<float>(direction[axis])));
    }
    auto inst = decoder->decode_window(rdna3_5::build_mimg(
        rdna3_5::kImageBvhIntersectRayMimg, {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = 16}));
    ASSERT_TRUE(inst.succeeded());
    ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    for (unsigned word = 0; word < 4; ++word)
      EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + word, 0), expected[word]);
  }
};

INSTANTIATE_TEST_SUITE_P(Gfx11, Gfx11BvhTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5));

TEST_P(Gfx11BvhTest, InvalidResultNullifiesBeforeNodeLookup) {
  cu_->write_sgpr(sb_, kBase >> 8);
  cu_->write_sgpr(sb_ + 2, 0); // The descriptor covers node zero.
  cu_->write_sgpr(sb_ + 3, kBvhTypeField);
  auto decoder = Decoder::create(arch());
  for (uint32_t dst : {28u, 29u}) {
    for (uint32_t pointer : {0u, 8u}) {
      for (uint64_t exec : {0u, 1u}) {
        SCOPED_TRACE(testing::Message()
                     << "dst=" << dst << " pointer=" << pointer << " exec=" << exec);
        wf_->set_exec(exec);
        cu_->write_vgpr(vb_, 0, pointer);
        const auto words =
            rdna3_5::build_mimg(rdna3_5::kImageBvhIntersectRayMimg,
                                {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = uint8_t(dst)});
        auto decoded = decoder->decode_window(words);
        ASSERT_TRUE(decoded.succeeded());
        auto inst = std::move(decoded).value();
        const auto issued = cu_->execute_instruction(inst.get(), *wf_);
        const uint64_t lanes = dst == 28 ? exec : 0;
        ASSERT_TRUE(issued.succeeded());
        const auto *state = inst->data_as<amdgpu::VectorMemState>();
        ASSERT_TRUE(state);
        EXPECT_EQ(state->exec_mask, lanes);
        EXPECT_EQ(state->lane_mask, pointer == 0 ? lanes : 0);
        EXPECT_EQ(state->destination_vgpr_count(), 4u);
        const auto *info = inst->amdgpu_memory_issue_info();
        ASSERT_TRUE(info && !info->empty());
        EXPECT_EQ(info->counter_obligations().front().wait_counter_type(),
                  amdgpu::WaitCounterType::LOADCNT);
        if (lanes && pointer) {
          amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
          ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
          for (unsigned word = 0; word < 4; ++word)
            EXPECT_EQ(cu_->read_vgpr(vb_ + dst + word, 0), 0xffffffffu);
        }
        EXPECT_EQ(wf_->exec(), exec);
      }
    }
  }
}

TEST_P(Gfx11BvhTest, NullifiedResultStillReadsThePendingDescriptor) {
  // An SMEM load produces one descriptor word. A VMEM load independently produces
  // the node pointer, which remains pending across s_waitcnt lgkmcnt(0).
  constexpr uint32_t data = 0x8000;
  constexpr std::array<uint32_t, 4> resource{kBase >> 8, 0, 0, kBvhTypeField};
  for (unsigned i = 0; i < resource.size(); ++i) {
    memory_.write32(data + i * 4, resource[i]);
    cu_->write_sgpr(sb_ + i, resource[i]);
  }
  memory_.write32(data + 16, 8); // Beyond the descriptor's last node.
  cu_->write_sgpr(sb_ + 8, data);
  cu_->write_sgpr(sb_ + 9, 0);
  cu_->write_vgpr(vb_ + 20, 0, data + 16);
  cu_->write_vgpr(vb_ + 21, 0, 0);
  wf_->set_exec(1);
  const auto pointer =
      rdna3::build_flat(rdna3::kFlatLoadB32Flat, {.seg = 2, .addr = 20, .saddr = 124, .vdst = 0});
  uint32_t pc = 0x1000;
  for (uint8_t word = 0; word < resource.size(); ++word) {
    const auto scalar = rdna3::build_smem(
        rdna3::kSLoadB32Smem, {.sbase = 4, .sdata = word, .offset = 4u * word, .soffset = 124});
    for (auto opcode : {rdna3::kImageBvhIntersectRayMimg, rdna3::kImageBvh64IntersectRayMimg}) {
      for (uint8_t dst : {28, 29}) {
        auto bvh = rdna3::build_mimg(opcode, {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = dst});
        for (bool waited : {false, true}) {
          SCOPED_TRACE(testing::Message() << "word=" << unsigned(word) << " opcode=" << opcode
                                          << " dst=" << unsigned(dst) << " waited=" << waited);
          const auto sync = waited ? rdna3::build_sopp(rdna3::kSWaitcntSopp, {.simm16 = 0xfc07})
                                   : rdna3::build_sopp(rdna3::kSNopSopp);
          const std::array code{scalar[0], scalar[1], pointer[0], pointer[1],
                                sync[0],   bvh[0],    bvh[1]};
          memory_.load_image(reinterpret_cast<const uint8_t *>(code.data()), sizeof(code), pc);
          wf_->pc = pc;
          pc += 64;
          wf_->ensure_memory_wait_scoreboard().clear();
          ASSERT_TRUE(cu_->step());
          ASSERT_TRUE(cu_->step());
          ASSERT_TRUE(cu_->step());
          const auto before = cu_->memory_wait_diagnostic_count();
          ASSERT_TRUE(cu_->step());
          EXPECT_EQ(cu_->memory_wait_diagnostic_count() - before,
                    unsigned(!waited) + unsigned(dst == 28));
        }
      }
    }
  }
}

class Gfx11ImageAtomicTest : public ImageAccessFixture,
                             public testing::WithParamInterface<rj_code_arch_t> {
protected:
  rj_code_arch_t arch() const override { return GetParam(); }
};
INSTANTIATE_TEST_SUITE_P(Gfx11, Gfx11ImageAtomicTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5));

TEST_P(Gfx11ImageAtomicTest, AtomicsRejectZeroDmask) { check_zero_dmask_atomics(); }

TEST_P(Gfx11ImageAtomicTest, UnboundAtomicsValidateTheEncodedPayload) {
  wf_->set_exec(1);
  for (unsigned i = 0; i < 8; ++i)
    cu_->write_sgpr(sb_ + i, 0);
  auto decoder = Decoder::create(arch());
  for (bool compare : {false, true}) {
    const auto opcode = compare ? rdna3_5::kImageAtomicCmpswapMimg : rdna3_5::kImageAtomicAddMimg;
    for (uint8_t mask = 0; mask < 16; ++mask) {
      SCOPED_TRACE(testing::Message() << "cmpswap=" << compare << " dmask=" << unsigned(mask));
      auto decoded = decoder->decode_window(rdna3_5::build_mimg(
          opcode, {.dim = 1, .unorm = 1, .dmask = mask, .glc = 1, .vdata = 8, .tfe = 1}));
      ASSERT_TRUE(decoded.succeeded());
      const bool legal = compare ? mask == 3 || mask == 15 : mask == 1 || mask == 3;
      wf_->clear_instruction_execution_error();
      auto &inst = decoded.value();
      const auto issued = cu_->execute_instruction(inst.get(), *wf_);
      ASSERT_EQ(issued.succeeded(), legal);
      if (legal) {
        ASSERT_NE(inst->data(), nullptr);
        EXPECT_EQ(inst->data_as<amdgpu::VectorMemState>()->lane_mask, 0u);
      } else {
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnsupportedOperandValue);
        EXPECT_EQ(inst->data(), nullptr);
      }
    }
  }
}

TEST_P(Gfx11ImageAtomicTest, AtomicAddRequiresUnnormalizedCoordinates) {
  const auto desc = rdna_image_descriptor(arch(), {.format = kR32Uint});
  for (unsigned i = 0; i < desc.size(); ++i)
    cu_->write_sgpr(sb_ + i, desc[i]);
  auto decoder = Decoder::create(arch());
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (bool returning : {false, true}) {
    for (bool unorm : {false, true}) {
      SCOPED_TRACE(testing::Message() << "unorm=" << unorm << " returning=" << returning);
      memory_.write32(kBase, 5);
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      cu_->write_vgpr(vb_ + 8, 0, 3);
      const auto words =
          rdna3_5::build_mimg(rdna3_5::kImageAtomicAddMimg,
                              {.dim = 1, .unorm = unorm, .dmask = 1, .glc = returning, .vdata = 8});
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      auto inst = std::move(decoded).value();
      const auto issued = cu_->execute_instruction(inst.get(), *wf_);
      if (!unorm) {
        EXPECT_TRUE(issued.failed());
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnsupportedOperandValue);
        EXPECT_EQ(inst->data(), nullptr);
      } else {
        ASSERT_TRUE(issued.succeeded());
        amdgpu::GlobalMemPipeline pipe(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipe.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      }
      EXPECT_EQ(memory_.read32(kBase), unorm ? 8u : 5u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), unorm && returning ? 5u : 3u);
    }
  }
}

TEST_P(Gfx11BvhTest, ReturnsHardwareRoundedProjectedBarycentrics) {
  // W7900 IMAGE_BVH_INTERSECT_RAY, tag 0, barycentric selectors 1 and 2.
  struct Case {
    std::array<uint32_t, 9> vertices;
    std::array<uint32_t, 3> origin, direction;
    std::array<uint32_t, 4> expected;
  };
  constexpr Case cases[]{
      {{0x40c88e04, 0xc0b760b3, 0xbfc5c580, 0xc0a232a0, 0x3f285210, 0x40a6ef68, 0xc0fd5513,
        0x3ff1a668, 0x402993f4},
       {0xc0c9b3cb, 0x40651057, 0x3d2c07a0},
       {0x40b0161c, 0xc0a534f0, 0x3f8313e8},
       {0xc57b32f7, 0xc55f49c1, 0xc45f49c0, 0xc45f49c7}},
      {{0xbf6f08f0, 0x40c32818, 0xbe2de200, 0xc0c9c1af, 0x400ef1d4, 0x3f9d2f98, 0xc023afa8,
        0xc0da6180, 0x3f032200},
       {0xc08239a4, 0x410a4fe5, 0x40a927b2},
       {0x3f9dd0b8, 0xc0bfcc58, 0xc08c6664},
       {0x46282fe8, 0x46157feb, 0x45157fef, 0x45157fe8}},
      {{0xc0bc788f, 0xbeeca3f0, 0x40749aa8, 0x40c081b4, 0x40cacb42, 0xc014036e, 0xc07a6d16,
        0xc0df8343, 0xc0566c70},
       {0x3fcc0db4, 0x3fc9dbd4, 0x40994258},
       {0xc0644fd8, 0xbfe023d8, 0xc0744a12},
       {0xc6094f66, 0xc5f41b60, 0xc4f41b61, 0xc4f41b61}},
  };
  for (const auto &test : cases)
    ASSERT_NO_FATAL_FAILURE(check_triangle(test.vertices, test.origin, test.direction, 10.f,
                                           /*barycentrics=*/true, test.expected));
}

TEST_P(Gfx11BvhTest, DominantAxisTiesMatchW7900BarycentricResults) {
  constexpr std::array<uint32_t, 9> vertices{0x408f96ec, 0x3f8f5768, 0x3f9adb10,
                                             0x40c35630, 0xc0ec332a, 0x40eae066,
                                             0x4036dc68, 0xbeef7d20, 0x40d14674};
  struct Case {
    std::array<uint32_t, 3> origin, direction;
    std::array<uint32_t, 4> expected;
  };
  // Each W7900 result differs from projecting on another tied axis.
  constexpr Case cases[]{
      // X/Y
      {{0x403999ff, 0xbfdbabdc, 0x4073b571},
       {0x4006091c, 0x4006091c, 0x3e97e750},
       {0x43920333, 0x4401c9f5, 0x4281c9f6, 0x4342aef1}},
      // X/-Y
      {{0x403999ff, 0x3f23d147, 0x4073b571},
       {0x4006091c, 0xc006091c, 0x3e97e750},
       {0x42c7649a, 0x43313cfa, 0x41b13cfb, 0x4284edbd}},
      // Y/Z
      {{0x407a50db, 0xbfdbabdc, 0x4032fe95},
       {0x3e97e750, 0x4006091c, 0x4006091c},
       {0x434f1209, 0x43b81006, 0x42381006, 0x430a0c05}},
      // Y/-Z
      {{0x407a50db, 0xbfdbabdc, 0x40a4e46a},
       {0x3e97e750, 0x4006091c, 0xc006091c},
       {0x421560df, 0x4284c7db, 0x4104c7d5, 0x41c72bc6}},
      // X/-Z
      {{0x403999ff, 0xbf347c46, 0x40a4e46a},
       {0x4006091c, 0x3e97e750, 0xc006091c},
       {0x42e1329a, 0x43482cf8, 0x41c82cf9, 0x429621b9}},
      // X/-Y/Z
      {{0x403999ff, 0x3f23d147, 0x4032fe95},
       {0x4006091c, 0xc006091c, 0x4006091c},
       {0x432c8997, 0x43995ddb, 0x42195ddc, 0x42e60ccb}},
  };
  for (const auto &test : cases)
    ASSERT_NO_FATAL_FAILURE(check_triangle(vertices, test.origin, test.direction, 10.f,
                                           /*barycentrics=*/true, test.expected));
}

TEST_P(Gfx11BvhTest, RoundedEdgesMatchW7900InBothResultModes) {
  using VectorBits = std::array<uint32_t, 3>;
  using Words = std::array<uint32_t, 4>;
  const auto check = [&](const std::array<uint32_t, 9> &vertices, const VectorBits &origin,
                         const VectorBits &direction, const Words &status,
                         const Words &barycentrics) {
    ASSERT_NO_FATAL_FAILURE(check_triangle(vertices, origin, direction, 100.f, false, status));
    ASSERT_NO_FATAL_FAILURE(check_triangle(vertices, origin, direction, 100.f, true, barycentrics));
  };
  constexpr std::array<uint32_t, 9> vertices{0x408f96ec, 0x3f8f5768, 0x3f9adb10,
                                             0x40c35630, 0xc0ec332a, 0x40eae066,
                                             0x4036dc68, 0xbeef7d20, 0x40d14674};
  constexpr VectorBits direction{0x4100574e, 0xc0653e8a, 0x4057c06c};
  constexpr Words miss{0x7f800000, 0x3f800000, 9, 0};
  // A negative FP32 J is a miss even though the FP64 projection puts the ray inside.
  ASSERT_NO_FATAL_FAILURE(check(vertices, {0xc02105b6, 0xbf269670, 0x3fd963e8}, direction, miss,
                                {0x7f800000, 0x4689c732, 0x462d7f78, 0xba00574e}));
  // The two zero-J captures land on opposite sides of the projected top-left ownership rule.
  ASSERT_NO_FATAL_FAILURE(check(vertices, {0xc02105b4, 0xbf26966b, 0x3fd963e5}, direction,
                                {0x4689c734, 0x4689c731, 9, 1},
                                {0x4689c734, 0x4689c731, 0x462d7f76, 0}));
  ASSERT_NO_FATAL_FAILURE(check({0x40e2de98, 0xbe344300, 0x3f894ea8, 0xc00ba9e4, 0x40ee6580,
                                 0x40df95f0, 0xc0a14f15, 0x403b2328, 0x40906512},
                                {0xc1b0d58b, 0xc0efacd3, 0xc0addf64},
                                {0x40ca4e3d, 0x40798bac, 0x40510a38}, miss,
                                {0xff800000, 0xc5806388, 0xc54dbb88, 0}));
}

TEST_P(Gfx11BvhTest, TriangleStatusUsesBasePrimitiveOrPerTagId) {
  const auto bits = [](float f) { return std::bit_cast<uint32_t>(f); };
  // These W7900 rays hit tags 0 and 1, and miss tags 2 and 3.
  const std::array<float, 9> vertices{0, 0, 1, 0, 1, 1, 1, 0, 1};
  for (unsigned i = 0; i < vertices.size(); ++i)
    memory_.write32(kBase + i * 4, bits(vertices[i]));
  auto decoder = Decoder::create(arch());
  cu_->write_sgpr(sb_, kBase >> 8);
  wf_->set_exec(15);
  for (unsigned tag = 0; tag < 4; ++tag) {
    cu_->write_vgpr(vb_, tag, tag);
    cu_->write_vgpr(vb_ + 2, tag, bits(20));
    const float data[]{.25f, .25f, 0.f, 0.f, 0.f, 1.f, INFINITY, INFINITY, 1.f};
    for (unsigned i = 0; i < 9; ++i)
      cu_->write_vgpr(vb_ + 3 + i, tag, bits(data[i]));
  }
  const auto instruction = rdna3_5::build_mimg(rdna3_5::kImageBvh64IntersectRayMimg,
                                               {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = 16});
  for (uint32_t base_id : {0x80ff12feu, 0xfffffffeu}) {
    memory_.write32(kBase + 60, base_id);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    for (bool pointer_flags : {false, true}) {
      cu_->write_sgpr(sb_ + 3, kBvhTypeField | (uint32_t(pointer_flags) << 23));
      for (bool pointer_flip : {false, true}) {
        if (!pointer_flags && pointer_flip)
          continue;
        for (unsigned tag = 0; tag < 4; ++tag)
          cu_->write_vgpr(vb_ + 1, tag, uint32_t(pointer_flip) << 25);
        auto decoded = decoder->decode_window(instruction);
        ASSERT_TRUE(decoded.succeeded());
        auto op = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(op.get(), *wf_).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(op.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
        for (unsigned tag = 0; tag < 4; ++tag) {
          SCOPED_TRACE(testing::Message()
                       << "base_id=0x" << std::hex << base_id << std::dec << " tag=" << tag
                       << " pointer_flags=" << pointer_flags << " pointer_flip=" << pointer_flip);
          const bool hit = tag < 2;
          const uint32_t id_byte = (base_id >> (8 * tag)) & 0xff;
          const uint32_t id = pointer_flags ? id_byte : base_id + tag;
          // Bit 5 of the selected ID reverses winding in pointer-flag mode.
          const bool flip = pointer_flip != (pointer_flags && (id_byte & 0x20));
          const float sign = (tag == 1) != flip ? -1.f : 1.f;
          const uint32_t numerator = bits(hit ? sign * (tag == 1 ? .5f : 1.f) : INFINITY);
          const std::array<uint32_t, 4> expected{numerator, bits(hit ? sign : 1.f), id, hit};
          for (unsigned i = 0; i < 4; ++i)
            EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + i, tag), expected[i]);
        }
      }
    }
  }
}

TEST_P(Gfx11BvhTest, FlaggedPointerWaitsReadRayOnlyWhenDescriptorEnablesFlags) {
  wf_->set_exec(1);
  cu_->write_sgpr(sb_ + 2, 0); // The low pointer selects the only in-bounds triangle.
  cu_->write_vgpr(vb_, 0, 0);
  cu_->write_vgpr(vb_ + 1, 0, kForceOpaque << 22);
  const auto words = rdna3_5::build_mimg(rdna3_5::kImageBvh64IntersectRayMimg,
                                         {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = 16});
  auto decoded = Decoder::create(arch())->decode_window(words);
  ASSERT_TRUE(decoded.succeeded());
  for (bool pointer_flags : {false, true}) {
    SCOPED_TRACE(testing::Message() << "pointer_flags=" << pointer_flags);
    cu_->write_sgpr(sb_ + 3, kBvhTypeField | (uint32_t(pointer_flags) << 23));
    for (unsigned source = 0; source < 12; ++source)
      expect_source_wait(*decoded.value(), 0, source, source < 2 || pointer_flags);
  }
}

TEST_P(Gfx11BvhTest, TriangleIdControlsFaceCullingAndWinding) {
  const auto bits = [](float f) { return std::bit_cast<uint32_t>(f); };
  // The selected ID byte varies bits 4 (double-sided) and 5 (flip) for each triangle tag.
  constexpr std::array<uint32_t, 4> ids{0x80, 0x90, 0xa0, 0xb0};
  constexpr std::array<float, 9> vertices{0, 0, 1, 0, 1, 1, 1, 0, 1};
  constexpr std::array<float, 10> ray{20, .25f, .25f, 0, 0, 0, 1, INFINITY, INFINITY, 1};
  constexpr unsigned lane_count = std::size(kTriangleVertices) * ids.size();
  std::array<uint32_t, lane_count> id_words;
  for (unsigned lane = 0; lane < lane_count; ++lane) {
    const unsigned tag = lane / ids.size();
    const uint32_t id = ids[lane % ids.size()];
    // For a nonzero tag, byte 0 carries the opposite double-sided bit.
    id_words[lane] = tag == 0 ? id : (id << (8 * tag)) | (id ^ 0x10u);
    const uint64_t address = kBase + 64 * lane;
    for (unsigned vertex = 0; vertex < 3; ++vertex)
      for (unsigned axis = 0; axis < 3; ++axis)
        memory_.write32(address + 12 * kTriangleVertices[tag][vertex] + 4 * axis,
                        bits(vertices[3 * vertex + axis]));
    memory_.write32(address + 60, id_words[lane]);
    cu_->write_vgpr(vb_, lane, 8 * lane + tag);
    for (unsigned i = 0; i < ray.size(); ++i)
      cu_->write_vgpr(vb_ + 2 + i, lane, bits(ray[i]));
  }
  wf_->set_exec((1u << lane_count) - 1);
  cu_->write_sgpr(sb_, kBase >> 8);
  cu_->write_sgpr(sb_ + 2, lane_count - 1);
  const auto words = rdna3_5::build_mimg(rdna3_5::kImageBvh64IntersectRayMimg,
                                         {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = 16});
  auto decoder = Decoder::create(arch());
  struct Case {
    bool pointer_flags;
    uint32_t flags;
    uint32_t hit_id_mask;
  };
  for (const auto [pointer_flags, flags, hit_id_mask] : {Case{false, 0, 0xf},
                                                         {true, kCullFront, 0xe},
                                                         {true, kCullBack, 0xb},
                                                         {true, kCullFront | kFlip, 0xb},
                                                         {true, kCullBack | kFlip, 0xe},
                                                         {true, kCullFront | kDoubleSided, 0xf},
                                                         {true, kCullBack | kDoubleSided, 0xf},
                                                         {true, kCullOpaque, 0}}) {
    for (unsigned lane = 0; lane < lane_count; ++lane)
      cu_->write_vgpr(vb_ + 1, lane, flags << 22);
    for (bool barycentrics : {false, true}) {
      cu_->write_sgpr(sb_ + 3, kBvhTypeField | (uint32_t(pointer_flags) << 23) |
                                   (uint32_t(barycentrics) << 24));
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      auto op = std::move(decoded).value();
      ASSERT_TRUE(cu_->execute_instruction(op.get(), *wf_).succeeded());
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipeline.issue(op.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      for (unsigned lane = 0; lane < lane_count; ++lane) {
        const unsigned tag = lane / ids.size();
        const unsigned id_index = lane % ids.size();
        const uint32_t id = ids[id_index];
        SCOPED_TRACE(testing::Message()
                     << "tag=" << tag << " id_word=0x" << std::hex << id_words[lane] << " flags=0x"
                     << flags << " pointer_flags=" << pointer_flags
                     << " barycentrics=" << barycentrics);
        const bool hit = hit_id_mask & (1u << id_index);
        const bool flip = bool(flags & kFlip) != (pointer_flags && (id & 0x20));
        const float sign = flip ? -1.f : 1.f;
        using Words = std::array<uint32_t, 4>;
        const Words expected = barycentrics
                                   ? Words{bits(hit ? sign : std::copysign(INFINITY, sign)),
                                           bits(sign), bits(.5f * sign), bits(.5f * sign)}
                                   : Words{bits(hit ? sign : INFINITY), bits(hit ? sign : 1.f),
                                           pointer_flags ? id : id_words[lane] + tag, hit};
        for (unsigned word = 0; word < expected.size(); ++word)
          EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + word, lane), expected[word]);
      }
    }
  }
}

TEST_P(Gfx11BvhTest, IntersectsMixedNodeTypesAtCompletion) {
  const auto bits = [](float f) { return std::bit_cast<uint32_t>(f); };
  constexpr uint32_t missing = 0xffffffff;
  auto decoder = Decoder::create(arch());
  for (uint64_t descriptor_base : {kBase, UINT64_C(0xffff800000010000)}) {
    for (bool wide : {false, true}) {
      // BVH32 adds a descriptor base. BVH64 uses raw addresses above 32 GB.
      if (wide && descriptor_base != kBase)
        continue;
      const uint64_t base = wide ? (uint64_t{1} << 36) + kBase : descriptor_base;
      const uint64_t node_base = wide ? base / 8 : 0;
      const uint64_t bound = node_base / 8 + 4;
      cu_->write_sgpr(sb_, wide ? 0u : uint32_t(base >> 8));
      const uint32_t base_high = wide ? 0u : uint32_t((base >> 40) & 255);
      cu_->write_sgpr(sb_ + 1, base_high);
      cu_->write_sgpr(sb_ + 2, uint32_t(bound));
      for (unsigned i = 0; i < 3; ++i)
        for (unsigned ax = 0; ax < 3; ++ax) {
          const float vertex = ax == 2 || (i == 1 && ax == 1) || (i == 2 && ax == 0) ? 1 : 0;
          memory_.write32(base + 4 * (i * 3 + ax), bits(vertex));
        }
      memory_.write32(base + 60, 0x12340009); // I=vertex 1, J=vertex 2.
      // Four children, one ray miss. FP32 / FP16 nodes use the same coordinates.
      const uint32_t child[4] = {0xaa, 0xbb, 0xcc, 0xdf};
      for (unsigned i = 0; i < 4; ++i) {
        memory_.write32(base + 64 + i * 4, child[i]);
        memory_.write32(base + 192 + i * 4, child[i]);
        const float near = i == 0 ? 5 : i + 1;
        const std::array<float, 6> bounds{i == 2 ? 2.f : -1.f, -1.f, near, 4.f, 4.f, near + 1};
        for (unsigned j = 0; j < 6; ++j)
          memory_.write32(base + 80 + (i * 6 + j) * 4, bits(bounds[j]));
        write_box16_bounds(base + 208 + i * 12, bounds);
      }
      memory_.write32(base + 64 + 28 * 4, 0x02000101); // Box32 child opacity flags.
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      for (bool a16 : {false, true}) {
        for (bool nsa : {false, true}) {
          const unsigned extent = nsa ? 3 : 1 + wide;
          const unsigned origin = nsa ? 6 : extent + 1;
          const unsigned dir = nsa ? 10 : origin + 3;
          const unsigned inv = nsa ? 13 : dir + 3;
          const auto encoding = rdna3_5::build_mimg(
              wide ? rdna3_5::kImageBvh64IntersectRayMimg : rdna3_5::kImageBvhIntersectRayMimg,
              {.nsa = nsa, .unorm = 1, .dmask = 15, .r128 = 1, .a16 = a16, .vdata = 16});
          const uint32_t inst[]{encoding[0], encoding[1],
                                3u | (6u << 8) | (10u << 16) | (13u << 24)};
          wf_->set_exec(
              0x7d); // Valid hits, a user node, an out-of-range triangle and a straddling box32.
          for (unsigned lane = 0; lane < 7; ++lane) {
            const uint64_t pointer = node_base + (lane == 2   ? 8 + 5
                                                  : lane == 3 ? 24 + 4
                                                  : lane == 4 ? 6
                                                  : lane == 5 ? 5 * 8
                                                  : lane == 6 ? 4 * 8 + 5
                                                              : 0);
            cu_->write_vgpr(vb_, lane, uint32_t(pointer));
            if (wide)
              cu_->write_vgpr(vb_ + 1, lane, pointer >> 32);
            cu_->write_vgpr(vb_ + extent, lane, bits(20));
            for (unsigned ax = 0; ax < 3; ++ax) {
              cu_->write_vgpr(vb_ + origin + ax, lane, bits(ax == 2 ? 0 : 0.25f));
              cu_->write_vgpr(vb_ + dir + ax, lane,
                              a16 ? (ax == 2 ? 0x3c003c00 : 0x7c000000) : bits(ax == 2 ? 1 : 0));
              if (!a16)
                cu_->write_vgpr(vb_ + inv + ax, lane, ax == 2 ? bits(1) : 0x7f800000);
            }
            for (unsigned j = 0; j < 4; ++j)
              cu_->write_vgpr(vb_ + 16 + j, lane, 0xdeadbeef);
          }
          cu_->write_sgpr(sb_ + 3, kBvhTypeField | uint32_t(bound >> 32));
          auto decoded = decoder->decode(inst);
          ASSERT_TRUE(decoded.succeeded());
          for (unsigned lane : {0u, 4u, 5u, 6u})
            for (unsigned source : {0u, extent})
              expect_source_wait(*decoded.value(), lane, source, source == 0 || lane == 0);
          for (auto [sort_boxes, return_barycentrics] :
               {std::pair{false, false}, {false, true}, {true, false}, {true, true}}) {
            cu_->write_sgpr(sb_ + 1, base_high | (uint32_t(sort_boxes) << 31));
            cu_->write_sgpr(sb_ + 3, kBvhTypeField | (uint32_t(return_barycentrics) << 24) |
                                         uint32_t(bound >> 32));
            auto decoded = decoder->decode_window(inst);
            ASSERT_TRUE(decoded.succeeded());
            auto op = std::move(decoded).value();
            ASSERT_NE(op->amdgpu_memory_issue_info(), nullptr);
            ASSERT_TRUE(cu_->execute_instruction(op.get(), *wf_).succeeded());
            const auto *state = op->data_as<amdgpu::VectorMemState>();
            ASSERT_NE(state, nullptr);
            EXPECT_EQ(state->lane_mask, 0xdu);
            ASSERT_EQ(state->element_lane_masks.size(), 2u);
            EXPECT_EQ(state->element_lane_masks[1], 4u);
            amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
            ASSERT_EQ(pipeline.issue(op.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
            const std::array<uint32_t, 4> tri{bits(1), bits(1),
                                              return_barycentrics ? bits(0.25f) : 0x12340009,
                                              return_barycentrics ? bits(0.25f) : 1u};
            const std::array<uint32_t, 4> box = sort_boxes
                                                    ? std::array{0xbbu, 0xdfu, 0xaau, missing}
                                                    : std::array{0xaau, 0xbbu, missing, 0xdfu};
            for (unsigned j = 0; j < 4; ++j) {
              EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 0), tri[j]);
              EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 1), 0xdeadbeef);
              EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 2), box[j]);
              EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 3), box[j]);
              for (unsigned lane : {4u, 5u, 6u})
                EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, lane), missing);
            }
          }
          if (!wide || a16 || nsa)
            continue;
          // W7900 pointer-flag controls. Box16 has no per-child opacity metadata.
          using Words = std::array<uint32_t, 4>;
          const Words box_hits{0xaa, 0xbb, missing, 0xdf},
              no_hits{missing, missing, missing, missing};
          const Words procedural{missing, missing, missing, 0xdf},
              triangles{0xaa, 0xbb, missing, missing};
          const Words tri_hit{bits(1), bits(1), 9, 1},
              tri_hit_bary{bits(1), bits(1), bits(.25f), bits(.25f)};
          const Words tri_miss{bits(INFINITY), bits(1), 9, 0};
          const Words tri_miss_bary{bits(INFINITY), bits(1), bits(.25f), bits(.25f)};
          struct FlagCase {
            uint32_t flags;
            Words box16, box32, triangle, triangle_bary;
            bool pointer_flags = true;
          };
          for (const FlagCase &test :
               {FlagCase{0, box_hits, box_hits, tri_hit, tri_hit_bary},
                // Any nonzero upper pointer flags reject the pointer when disabled.
                FlagCase{kForceOpaque, no_hits, no_hits, no_hits, no_hits, false},
                FlagCase{kForceOpaque | kCullOpaque, no_hits, no_hits, no_hits, no_hits},
                FlagCase{kForceNonOpaque | kCullNonOpaque, no_hits, no_hits, no_hits, no_hits},
                FlagCase{kForceNonOpaque | kCullOpaque, box_hits, box_hits, tri_hit, tri_hit_bary},
                FlagCase{kForceOpaque | kCullNonOpaque, box_hits, box_hits, tri_hit, tri_hit_bary},
                FlagCase{kCullOpaque | kCullNonOpaque, no_hits, no_hits, no_hits, no_hits},
                FlagCase{kSkipTriangles, procedural, procedural, no_hits, no_hits},
                FlagCase{kSkipProcedural, triangles, triangles, tri_hit, tri_hit_bary},
                FlagCase{kCullOpaque, box_hits, procedural, tri_hit, tri_hit_bary},
                FlagCase{kCullNonOpaque, box_hits, triangles, tri_miss, tri_miss_bary},
                // Both force flags leave opacity unchanged.
                FlagCase{kForceOpaque | kForceNonOpaque | kCullOpaque, box_hits, procedural,
                         tri_hit, tri_hit_bary},
                FlagCase{kForceOpaque | kForceNonOpaque | kCullNonOpaque, box_hits, triangles,
                         tri_miss, tri_miss_bary}}) {
            SCOPED_TRACE(testing::Message() << "flags=0x" << std::hex << test.flags
                                            << " pointer_flags=" << test.pointer_flags);
            wf_->set_exec(0xd); // Triangle, Box32 and Box16 occupy lanes 0, 2 and 3.
            for (unsigned lane : {0u, 2u, 3u})
              cu_->write_vgpr(vb_ + 1, lane, uint32_t(node_base >> 32) | (test.flags << 22));
            cu_->write_sgpr(sb_ + 1, base_high);
            for (bool barycentrics : {false, true}) {
              cu_->write_sgpr(sb_ + 3, kBvhTypeField | (uint32_t(test.pointer_flags) << 23) |
                                           (uint32_t(barycentrics) << 24) | uint32_t(bound >> 32));
              auto decoded = decoder->decode_window(inst);
              ASSERT_TRUE(decoded.succeeded());
              auto op = std::move(decoded).value();
              ASSERT_TRUE(cu_->execute_instruction(op.get(), *wf_).succeeded());
              amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
              ASSERT_EQ(pipeline.issue(op.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
              for (unsigned j = 0; j < 4; ++j) {
                EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 0),
                          barycentrics ? test.triangle_bary[j] : test.triangle[j]);
                EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 2), test.box32[j]);
                EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 3), test.box16[j]);
              }
            }
          }
        }
      }
    }
  }
}

TEST_P(Gfx11BvhTest, Box32ChildMetadataCullsBoxAndInstancePointers) {
  constexpr uint32_t invalid = 0xffffffff;
  constexpr std::array<uint32_t, 4> children{0xac, 0xad, 0xae, 0xaf};
  for (unsigned child = 0; child < children.size(); ++child) {
    // Every child covers the ray. Pointer tags 4..7 select box, box, instance and procedural.
    memory_.write32(kBase + 4 * child, children[child]);
    memory_.write32(kBase + 128 + 4 * child, children[child]);
    const std::array<float, 6> bounds{-1, -1, 1, 4, 4, 2};
    for (unsigned j = 0; j < bounds.size(); ++j)
      memory_.write32(kBase + 16 + 4 * (6 * child + j), std::bit_cast<uint32_t>(bounds[j]));
    write_box16_bounds(kBase + 144 + child * 12, bounds);
  }
  memory_.write32(kBase + 28 * 4, 0x08040804); // Alternating triangle-only and procedural-only.
  cu_->write_sgpr(sb_, kBase >> 8);
  cu_->write_sgpr(sb_ + 2, 2);
  cu_->write_sgpr(sb_ + 3, kBvhTypeField | (1u << 23));
  wf_->set_exec(3);
  for (unsigned lane = 0; lane < 2; ++lane) {
    cu_->write_vgpr(vb_, lane, lane == 0 ? 5 : 16 + 4); // Box32 at +0, Box16 at +128.
    constexpr std::array<float, 10> ray{20, .25f, .25f, 0, 0, 0, 1, INFINITY, INFINITY, 1};
    for (unsigned j = 0; j < ray.size(); ++j)
      cu_->write_vgpr(vb_ + 2 + j, lane, std::bit_cast<uint32_t>(ray[j]));
  }
  auto decoder = Decoder::create(arch());
  const auto encoding = rdna3_5::build_mimg(rdna3_5::kImageBvh64IntersectRayMimg,
                                            {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = 16});
  using Words = std::array<uint32_t, 4>;
  for (bool skip_triangles : {false, true}) {
    const uint32_t flags = skip_triangles ? kSkipTriangles : kSkipProcedural;
    SCOPED_TRACE(testing::Message()
                 << "skip_triangles=" << skip_triangles << " flags=0x" << std::hex << flags);
    for (unsigned lane = 0; lane < 2; ++lane)
      cu_->write_vgpr(vb_ + 1, lane, flags << 22);
    auto decoded = decoder->decode_window(encoding);
    ASSERT_TRUE(decoded.succeeded());
    auto op = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(op.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(op.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    // W7900 returns these four words with the same child tags and metadata.
    const Words box32 =
        skip_triangles ? Words{invalid, 0xad, invalid, 0xaf} : Words{0xac, invalid, 0xae, invalid};
    const Words box16 = skip_triangles ? children : Words{0xac, 0xad, 0xae, invalid};
    for (unsigned j = 0; j < children.size(); ++j) {
      EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 0), box32[j]);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 16 + j, 1), box16[j]);
    }
  }
}

TEST_P(Gfx11BvhTest, AssignsSharedEdgesAndVerticesInRaySpace) {
  const auto bits = [](float value) { return std::bit_cast<uint32_t>(value); };
  // Physical gfx1100 hits for the three edges, three vertices and one interior point.
  const std::array<std::array<float, 2>, 7> points{
      {{.5f, 0}, {0, .5f}, {.5f, .5f}, {0, 0}, {1, 0}, {0, 1}, {.25f, .25f}}};
  const auto decoder = Decoder::create(arch());
  cu_->write_sgpr(sb_, kBase >> 8);
  cu_->write_sgpr(sb_ + 3, kBvhTypeField);
  memory_.write32(kBase + 60, 9);
  wf_->set_exec((1u << points.size()) - 1);
  for (unsigned axis = 0; axis < 3; ++axis) {
    const unsigned first_axis = (axis + 1) % 3, second_axis = (axis + 2) % 3;
    for (bool reverse : {false, true}) {
      for (unsigned vertex = 0; vertex < 3; ++vertex) {
        const auto address = kBase + 12 * vertex;
        memory_.write32(address + 4 * first_axis, bits(vertex == (reverse ? 2 : 1)));
        memory_.write32(address + 4 * second_axis, bits(vertex == (reverse ? 1 : 2)));
        memory_.write32(address + 4 * axis, bits(1));
      }
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      for (bool back : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "axis=" << axis << " reverse=" << reverse << " back=" << back);
        const float direction = back ? -1.f : 1.f;
        for (unsigned lane = 0; lane < points.size(); ++lane) {
          cu_->write_vgpr(vb_, lane, 0);
          cu_->write_vgpr(vb_ + 1, lane, bits(10));
          for (unsigned dim = 0; dim < 3; ++dim) {
            const float origin = dim == axis ? 1 - direction : points[lane][dim == second_axis];
            cu_->write_vgpr(vb_ + 2 + dim, lane, bits(origin));
            cu_->write_vgpr(vb_ + 5 + dim, lane, bits(dim == axis ? direction : 0.f));
            cu_->write_vgpr(vb_ + 8 + dim, lane, dim == axis ? bits(direction) : 0x7f800000);
          }
        }
        const auto words = rdna3_5::build_mimg(rdna3_5::kImageBvhIntersectRayMimg,
                                               {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = 16});
        auto decoded = decoder->decode_window(words);
        ASSERT_TRUE(decoded.succeeded());
        auto inst = std::move(decoded).value();
        ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
        const unsigned hit_lanes = back ? 0b1001011 : 0b1000010;
        for (unsigned lane = 0; lane < points.size(); ++lane)
          EXPECT_EQ(cu_->read_vgpr(vb_ + 19, lane), (hit_lanes >> lane) & 1u) << lane;
      }
    }
  }
}

TEST_P(Gfx11BvhTest, OppositeWindingsShareEdgesForNonUnitRayDirections) {
  using Vec = std::array<float, 3>;
  struct Case {
    Vec origin, direction, edge_start, edge_end, left, right;
    unsigned hit;
    uint32_t denominator;
  };
  // Physical gfx1100 raw hit records for two shared edges. Each ray hits at t=1.
  const Case cases[] = {
      {{-1.5f, -5.5f, -2},
       {4, 5, 4},
       {3.5f, -1, 4},
       {-.5f, 1, -4},
       {3.5f, -1, -3.5f},
       {-3, -3.5f, -1.5f},
       1,
       0x45386000},
      {{-1, 2.75f, 0},
       {-2, -3, -1},
       {-4, 3, -1.5f},
       {-2, -3.5f, -.5f},
       {-2, 3, 1},
       {-.5f, 4, 0},
       0,
       0xc3804000},
  };
  const auto decoder = Decoder::create(arch());
  wf_->set_exec(3);
  cu_->write_sgpr(sb_, kBase >> 8);
  cu_->write_sgpr(sb_ + 2, 1);
  cu_->write_sgpr(sb_ + 3, kBvhTypeField);
  for (const auto &ray : cases) {
    for (bool reverse : {false, true}) {
      for (unsigned lane = 0; lane < 2; ++lane) {
        const uint64_t node = kBase + 64 * lane;
        const bool swap = bool(lane) != reverse;
        const Vec vertices[] = {swap ? ray.edge_end : ray.edge_start,
                                swap ? ray.edge_start : ray.edge_end, lane ? ray.right : ray.left};
        cu_->write_vgpr(vb_, lane, lane << 3);
        cu_->write_vgpr(vb_ + 1, lane, std::bit_cast<uint32_t>(10.f));
        for (unsigned axis = 0; axis < 3; ++axis) {
          for (unsigned vertex = 0; vertex < 3; ++vertex)
            memory_.write32(node + 12 * vertex + 4 * axis,
                            std::bit_cast<uint32_t>(vertices[vertex][axis]));
          cu_->write_vgpr(vb_ + 2 + axis, lane, std::bit_cast<uint32_t>(ray.origin[axis]));
          cu_->write_vgpr(vb_ + 5 + axis, lane, std::bit_cast<uint32_t>(ray.direction[axis]));
          cu_->write_vgpr(vb_ + 8 + axis, lane, std::bit_cast<uint32_t>(1.f / ray.direction[axis]));
        }
        memory_.write32(node + 60, 9);
      }
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      auto inst = decoder->decode_window(rdna3_5::build_mimg(
          rdna3_5::kImageBvhIntersectRayMimg, {.unorm = 1, .dmask = 15, .r128 = 1, .vdata = 16}));
      ASSERT_TRUE(inst.succeeded());
      ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipeline.issue(inst.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      const uint32_t denominator = ray.denominator ^ (reverse ? 0x80000000u : 0u);
      for (unsigned lane = 0; lane < 2; ++lane) {
        const bool hit = ray.hit == lane;
        EXPECT_EQ(cu_->read_vgpr(vb_ + 16, lane), hit ? denominator : 0x7f800000u);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 17, lane), hit ? denominator : 0x3f800000u);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 18, lane), 9u);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 19, lane), unsigned(hit));
      }
    }
  }
}

TEST_P(Gfx11BvhTest, RotatesBarycentricsAndCullsTaggedTriangles) {
  const auto bits = [](float f) { return std::bit_cast<uint32_t>(f); };
  // All four tags select the same plane with different physical vertex sets.
  // W7900 selects vertex 0 for selector 3 in either returned barycentric word.
  constexpr std::array<float, 4> weights{.625f, .25f, .125f, .625f};
  for (unsigned tag = 0; tag < 4; ++tag) {
    const uint64_t address = kBase + tag * 64;
    for (unsigned i = 0; i < 3; ++i) {
      const auto index = kTriangleVertices[tag][i];
      memory_.write32(address + 12 * index, bits(i == 2 ? 1 : 0));
      memory_.write32(address + 12 * index + 4, bits(i == 1 ? 1 : 0));
      memory_.write32(address + 12 * index + 8, bits(1));
    }
    // Distinct selectors for I/J across the lanes. All triangles are opaque.
    memory_.write32(address + 60, (0x80u | tag | ((3u - tag) << 2)) << (tag * 8));
  }
  cu_->write_sgpr(sb_, kBase >> 8);
  cu_->write_sgpr(sb_ + 1, 0);
  cu_->write_sgpr(sb_ + 2, 3);
  auto decoder = Decoder::create(arch());
  wf_->set_exec(15);
  struct PointerMode {
    bool enabled;
    uint32_t flags;
  };
  for (const auto [enabled, flags] : {PointerMode{false, 0},
                                      {true, 0},
                                      {true, kFlip},
                                      {true, kFlip | kCullBack},
                                      {true, kCullOpaque},
                                      {true, kForceNonOpaque | kCullOpaque},
                                      {true, kCullFront},
                                      {true, kCullFront | kDoubleSided}}) {
    SCOPED_TRACE(testing::Message()
                 << "pointer_flags=" << enabled << " flags=0x" << std::hex << flags);
    cu_->write_sgpr(sb_ + 3, kBvhTypeField | (1u << 24) | (uint32_t(enabled) << 23));
    for (unsigned lane = 0; lane < 4; ++lane) {
      cu_->write_vgpr(vb_, lane, lane * 8 + lane);
      cu_->write_vgpr(vb_ + 1, lane, flags << 22);
      cu_->write_vgpr(vb_ + 2, lane, bits(10));
      cu_->write_vgpr(vb_ + 3, lane, bits(.125f));
      cu_->write_vgpr(vb_ + 4, lane, bits(.25f));
      cu_->write_vgpr(vb_ + 5, lane, 0);
      cu_->write_vgpr(vb_ + 6, lane, 0x7c000000);
      cu_->write_vgpr(vb_ + 7, lane, 0x7c000000);
      cu_->write_vgpr(vb_ + 8, lane, 0x3c003c00);
    }
    const auto words =
        rdna3_5::build_mimg(rdna3_5::kImageBvh64IntersectRayMimg,
                            {.unorm = 1, .dmask = 15, .r128 = 1, .a16 = 1, .vdata = 16});
    auto decoded = decoder->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    auto op = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(op.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(op.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    const bool miss = flags == kCullOpaque || flags == kCullFront || flags == (kFlip | kCullBack);
    const float sign = flags & kFlip ? -1.f : 1.f;
    for (unsigned lane = 0; lane < 4; ++lane) {
      EXPECT_EQ(cu_->read_vgpr(vb_ + 16, lane), bits(miss ? std::copysign(INFINITY, sign) : sign));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 17, lane), bits(sign));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 18, lane), bits(sign * weights[lane]));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 19, lane), bits(sign * weights[3 - lane]));
    }
  }
}

class Rdna4ImageMemoryTest : public ImageAccessFixture {
protected:
  rj_code_arch_t arch() const override { return ROCJITSU_CODE_ARCH_RDNA4; }

  void bind(uint32_t format, uint32_t selectors = kIdentity) {
    wf_->set_exec(1);
    const auto words = rdna_image_descriptor(arch(), {.format = format, .selectors = selectors});
    for (unsigned i = 0; i < words.size(); ++i)
      cu_->write_sgpr(sb_ + i, words[i]);
    write_coordinates(0, {0, 0, 0});
  }

  void issue(uint16_t opcode, uint8_t mask, bool returns = false) {
    const auto words = rdna4::build_vimage(
        opcode, {.dim = 1, .dmask = mask, .vdata = 8, .th = uint8_t(returns), .vaddr1 = 1});
    auto decoded = Decoder::create(arch())->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded()) << inst->mnemonic();
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
  }
};

TEST_F(Rdna4ImageMemoryTest, SampleBoundsAndDroppedFetchQualifyInputsBeforeIssue) {
  class Reads final : public ExecutionPlugin {
  public:
    Reads() : ExecutionPlugin("sample_coordinate_reads") {}
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t) override {
      if (reg == wave->vgpr_alloc().base)
        v0 |= lanes;
    }
    void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *, RegisterRef reg) override {
      if (reg.cls == RegClass::SGPR && reg.index < 12)
        scalar_reads |= 1u << reg.index;
    }
    uint32_t scalar_reads = 0;
    uint64_t v0 = 0;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Reads>();
  auto *reads = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  const auto resource =
      rdna_image_descriptor(arch(), {.format = 22, .width = 256, .height = 64, .pitch = 256});
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  cu_->write_sgpr(sb_ + 8, 2 | (2 << 3) | (2 << 6));
  cu_->write_sgpr(sb_ + 9, (3 * 256) << 13);
  cu_->write_sgpr(sb_ + 10, 1u << 26);
  cu_->write_sgpr(sb_ + 11, 0);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 0x3f800000);
  struct Case {
    uint8_t mask;
    bool tfe;
    uint8_t dst;
    bool reads_coordinates;
    bool writes_data;
    bool lwe = false;
  };
  for (auto c : {Case{1, true, 31, false, false}, Case{1, true, 30, true, true},
                 Case{1, false, 31, true, true}, Case{3, false, 31, false, false},
                 Case{0, false, 30, false, true}, Case{0, true, 30, false, true},
                 Case{0, true, 31, false, false}, Case{0, false, 30, true, true, true},
                 Case{0, false, 31, false, false, true}}) {
    SCOPED_TRACE(testing::Message() << unsigned(c.mask) << ' ' << c.tfe << ' ' << unsigned(c.dst));
    rdna4::VsampleBuilderFields fields{};
    fields.dim = 1;
    fields.dmask = c.mask;
    fields.tfe = c.tfe;
    fields.lwe = c.lwe;
    fields.unorm = true;
    fields.vdata = c.dst;
    fields.samp = 8;
    fields.vaddr1 = 1;
    fields.vaddr2 = 2;
    const auto words = rdna4::build_vsample(rdna4::kImageSampleLVsample, fields);
    auto decoded = Decoder::create(arch())->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
    scoreboard.clear();
    const auto kind = amdgpu::WaitCounterKind::Ds;
    scoreboard.add({scoreboard.issue(kind), 0x100, 1, {RegClass::VGPR, 0, 1}, kind, 0xf});
    scoreboard.add({scoreboard.issue(kind), 0x110, 1, {RegClass::SGPR, 0, 1}, kind, 0xf});
    scoreboard.add({scoreboard.issue(kind), 0x120, 1, {RegClass::VGPR, c.dst, 1}, kind, 0xf});
    std::array<unsigned, 2> warnings{};
    scoreboard.bind(0x200, &warnings, [](void *p, const auto &hazard) {
      ++(*static_cast<std::array<unsigned, 2> *>(p))[hazard.write];
    });
    scoreboard.check_instruction(*inst, *wf_);
    EXPECT_EQ(warnings[0], 2u * unsigned(c.reads_coordinates));
    EXPECT_EQ(warnings[1], unsigned(c.writes_data));
    cu_->write_vgpr(vb_ + c.dst, 0, 0xdeadbeef);
    if (c.tfe && c.dst < 31)
      cu_->write_vgpr(vb_ + c.dst + 1, 0, 0xcafe1234);
    reads->v0 = 0;
    reads->scalar_reads = 0;
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    EXPECT_EQ(reads->v0 != 0, c.reads_coordinates);
    EXPECT_EQ(reads->scalar_reads, c.reads_coordinates ? 0xfffu : 0u);
    const auto *state = inst->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->exec_mask, unsigned(c.writes_data));
    EXPECT_EQ(state->lane_mask, unsigned(c.mask && c.reads_coordinates));
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    if (!c.mask || !c.writes_data) {
      EXPECT_EQ(cu_->read_vgpr(vb_ + c.dst, 0), c.writes_data ? 0u : 0xdeadbeefu);
    }
    if (c.tfe && c.dst < 31) {
      EXPECT_EQ(cu_->read_vgpr(vb_ + c.dst + 1, 0), 0xcafe1234u);
    }
  }
}

TEST_F(Rdna4ImageMemoryTest, WaitsReadBoundVimageAndVsampleCoordinatesWithVgprZeroFallback) {
  const auto decoder = Decoder::create(arch());
  auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
  for (bool sample : {false, true}) {
    for (uint8_t y : {30, 32}) {
      const auto words =
          sample
              ? rdna4::build_vsample(
                    rdna4::kImageSampleLzVsample,
                    {.dim = 1, .dmask = 1, .vdata = 8, .samp = 8, .vaddr0 = 31, .vaddr1 = y})
              : rdna4::build_vimage(rdna4::kImageLoadVimage,
                                    {.dim = 1, .dmask = 1, .vdata = 8, .vaddr0 = 31, .vaddr1 = y});
      auto inst = decoder->decode_window(words);
      ASSERT_TRUE(inst.succeeded());
      for (uint32_t format : {0, 1}) {
        // R8_UNORM puts the only format bit below the GFX11 format field.
        bind(format);
        for (uint16_t reg : {0, 30, 31, 32}) {
          SCOPED_TRACE(testing::Message() << sample << ' ' << +y << ' ' << format << ' ' << reg);
          unsigned reports = 0;
          scoreboard.clear();
          const auto counter = amdgpu::WaitCounterKind::Load;
          scoreboard.add(
              {scoreboard.issue(counter), 0x100, 1, {RegClass::VGPR, reg, 1}, counter, 0xf});
          scoreboard.bind(0x200, &reports,
                          [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          scoreboard.check_instruction(*inst.value(), *wf_);
          EXPECT_EQ(reports, unsigned(format && (reg == 31 || reg == (y == 32 ? 0 : y))));
        }
      }
    }
  }
}

TEST_F(Rdna4ImageMemoryTest, PackedTransfersIgnoreConversionAndSelectors) {
  // An R8_UNORM view with all-zero channel selectors. PCK selects unconverted bytes.
  bind(1, 0);
  memory_.write32(kBase, 0x123456ab);
  issue(rdna4::kImageLoadPckVimage, 1);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabu);
  issue(rdna4::kImageLoadPckSgnVimage, 1);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xffffffabu);
  cu_->write_vgpr(vb_ + 8, 0, 0x135);
  issue(rdna4::kImageStorePckVimage, 1);
  EXPECT_EQ(memory_.read32(kBase), 0x12345635u);
}

TEST_F(Rdna4ImageMemoryTest, AtomicsRejectZeroDmask) { check_zero_dmask_atomics(); }

TEST_F(Rdna4ImageMemoryTest, ReturningAtomicsAcquireSourcesOnlyWhenTheirResultFits) {
  class SourceObserver final : public ExecutionPlugin {
  public:
    SourceObserver() : ExecutionPlugin("atomic_sources") {}
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t) override {
      if (lanes & 1)
        regs |= 1ull << (reg - wave->vgpr_alloc().base);
    }
    void onAmdgpuReadSgpr(const amdgpu::Wavefront *wave, uint32_t reg) override {
      const uint32_t word = reg - wave->sgpr_alloc().base;
      if (word < 8)
        descriptor_words |= 1u << word;
    }
    unsigned descriptor_words = 0;
    uint64_t regs = 0;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<SourceObserver>();
  auto *inputs = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  uint64_t pc = 0x40000;
  const auto scalar_load =
      rdna4::build_smem(rdna4::kSLoadB32Smem, {.sbase = 4, .sdata = 1, .soffset = 124});
  constexpr uint32_t descriptor_source = 0x50000;
  cu_->write_sgpr(sb_ + 8, descriptor_source);
  cu_->write_sgpr(sb_ + 9, 0);
  const auto load = rdna4::build_vds(rdna4::kDsLoadB32Vds, {.addr = 20, .vdst = 4});
  wf_->set_lds_size(64);
  wf_->lds().write32(0, 0);
  cu_->write_vgpr(vb_ + 20, 0, 0);
  auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
  for (const bool returns : {true, false}) {
    for (const uint8_t data : {30, 31}) {
      SCOPED_TRACE(testing::Message() << "return=" << returns << " vdata=" << +data);
      bind(48); // R32G32_UINT
      memory_.write32(descriptor_source, cu_->read_sgpr(sb_ + 1));
      constexpr uint64_t old = 0x0000002900000029ull;
      memory_.write64(kBase, old);
      cu_->write_vgpr(vb_, 0, 41);
      cu_->write_vgpr(vb_ + data, 0, 99);
      if (data == 30)
        cu_->write_vgpr(vb_ + 31, 0, 0);
      cu_->write_vgpr(vb_ + 4, 0, 0);
      cu_->write_vgpr(vb_ + 5, 0, 0);
      const auto words = rdna4::build_vimage(
          rdna4::kImageAtomicCmpswapVimage,
          {.dim = 1, .dmask = 15, .vdata = data, .th = uint8_t(returns), .vaddr0 = 4, .vaddr1 = 5});
      const bool suppressed = returns && data == 31;
      auto decoder = Decoder::create(arch());
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      auto &inst = *decoded.value();
      ASSERT_TRUE(cu_->execute_instruction(&inst, *wf_).succeeded());
      const auto *request = static_cast<const amdgpu::VectorMemState *>(inst.data());
      ASSERT_TRUE(request);
      EXPECT_EQ(request->exec_mask, uint64_t{!suppressed});
      EXPECT_EQ(request->lane_mask, uint64_t{!suppressed});
      EXPECT_EQ(request->wait_counter_type,
                returns ? amdgpu::WaitCounterType::LOADCNT : amdgpu::WaitCounterType::STORECNT);
      const uint64_t ds_pc = pc + sizeof(scalar_load);
      const uint64_t atomic_pc = ds_pc + sizeof(load);
      memory_.load_image(reinterpret_cast<const uint8_t *>(scalar_load.data()), sizeof(scalar_load),
                         pc);
      memory_.load_image(reinterpret_cast<const uint8_t *>(load.data()), sizeof(load), ds_pc);
      memory_.load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), atomic_pc);
      wf_->pc = pc;
      pc += 64;
      scoreboard.clear();
      cu_->step();
      ASSERT_EQ(wf_->pc, ds_pc);
      cu_->step();
      ASSERT_EQ(wf_->pc, atomic_pc);
      const auto before = cu_->memory_wait_diagnostic_count();
      inputs->descriptor_words = 0;
      inputs->regs = 0;
      cu_->step();
      ASSERT_FALSE(wf_->is_halted());
      cu_->l1_vector().flush_all();
      l2_.flush_all();
      const uint64_t expected_regs = suppressed ? 0
                                                : (1ull << 4) | (1ull << 5) | 1ull | (1ull << 31) |
                                                      (data == 30 ? 1ull << 30 : 0ull);
      EXPECT_EQ(inputs->regs, expected_regs);
      EXPECT_EQ(inputs->descriptor_words, 0xffu);
      // Every issue reads the pending s1 descriptor word. Live vector requests also consume v4.
      EXPECT_EQ(cu_->memory_wait_diagnostic_count() - before, 1u + unsigned(!suppressed));
      const uint64_t input = data == 30 ? 99 : 0x0000002900000063ull;
      EXPECT_EQ(memory_.read64(kBase), suppressed ? old : input);
    }
  }
}

TEST_F(Rdna4ImageMemoryTest, PayloadReadsAndWaitsSubstituteVgprZeroForEachMissingWord) {
  class Inputs final : public ExecutionPlugin {
  public:
    Inputs() : ExecutionPlugin("image_payload_inputs") {}
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t bytes) override {
      if (reg == wave->vgpr_alloc().base && (lanes & 1))
        v0_bytes |= bytes;
    }
    uint8_t v0_bytes = 0;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Inputs>();
  auto *inputs = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
  uint64_t pc = 0x40000;
  enum class Reach { Bound, Outside, Unbound };
  for (const auto op :
       {rdna4::kImageStoreVimage, rdna4::kImageStorePckVimage, rdna4::kImageAtomicCmpswapVimage}) {
    const bool atomic = op == rdna4::kImageAtomicCmpswapVimage;
    for (uint8_t data : {29u, 31u}) {
      for (auto reach : {Reach::Bound, Reach::Outside, Reach::Unbound}) {
        const bool in_bounds = reach == Reach::Bound;
        SCOPED_TRACE(testing::Message()
                     << +op << " v" << +data << " reach=" << static_cast<int>(reach));
        bind(reach == Reach::Unbound ? 0 : atomic ? kR32Uint : kRgba32Uint);
        const uint32_t first = atomic ? 99u : 0x11112222u;
        const uint32_t second = atomic ? 41u : 0x33334444u;
        cu_->write_vgpr(vb_, 0, second);
        cu_->write_vgpr(vb_ + data, 0, first);
        if (data == 29) {
          cu_->write_vgpr(vb_ + 30, 0, second);
          cu_->write_vgpr(vb_ + 31, 0, 0x55556666);
        }
        cu_->write_vgpr(vb_ + 4, 0, reach == Reach::Outside ? UINT32_MAX : 0);
        cu_->write_vgpr(vb_ + 5, 0, 0);
        memory_.write32(kBase, 41);
        memory_.write32(kBase + 4, 0);
        memory_.write32(kBase + 8, 0);
        const auto words = rdna4::build_vimage(op, {.dim = 1,
                                                    .dmask = uint8_t(atomic ? 3 : 7),
                                                    .vdata = data,
                                                    .th = uint8_t(atomic),
                                                    .vaddr0 = 4,
                                                    .vaddr1 = 5});
        memory_.load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), pc);
        wf_->pc = pc;
        pc += 0x40;
        scoreboard.clear();
        const auto counter = amdgpu::WaitCounterKind::Load;
        scoreboard.add({scoreboard.issue(counter), 0x100, 1, {RegClass::VGPR, 0, 1}, counter, 0xf});
        const auto warnings = cu_->memory_wait_diagnostic_count();
        inputs->v0_bytes = 0;
        cu_->run_issue_cycle(1);
        for (unsigned cycle = 0; cycle < 64 && !wf_->wait_counters().empty(); ++cycle)
          cu_->run_issue_cycle(0);
        ASSERT_TRUE(wf_->wait_counters().empty());
        cu_->l1_vector().flush_all();
        l2_.flush_all();
        const bool reads_v0 = in_bounds && data == 31;
        EXPECT_EQ(inputs->v0_bytes, reads_v0 ? 0xf : 0);
        EXPECT_EQ(cu_->memory_wait_diagnostic_count() - warnings, unsigned(reads_v0));
        EXPECT_EQ(memory_.read32(kBase), in_bounds ? first : 41u);
        if (!atomic) {
          EXPECT_EQ(memory_.read32(kBase + 4), in_bounds ? second : 0u);
          const auto third = data == 31 ? second : 0x55556666u;
          EXPECT_EQ(memory_.read32(kBase + 8), in_bounds ? third : 0u);
        } else if (in_bounds) {
          EXPECT_EQ(cu_->read_vgpr(vb_ + data, 0), 41u);
        }
      }
    }
  }
  // An odd D16 tuple uses only the low half of its last VGPR, including a substituted VGPR0.
  for (uint8_t pending_bytes : {0x3, 0xc}) {
    bind(kRgba32Uint);
    cu_->write_vgpr(vb_, 0, 0xdead3456);
    cu_->write_vgpr(vb_ + 4, 0, 0);
    cu_->write_vgpr(vb_ + 5, 0, 0);
    const auto words = rdna4::build_vimage(
        rdna4::kImageStoreVimage,
        {.dim = 1, .d16 = 1, .dmask = 1, .vdata = 32, .vaddr0 = 4, .vaddr1 = 5});
    memory_.load_image(reinterpret_cast<const uint8_t *>(words.data()), sizeof(words), pc);
    wf_->pc = pc;
    pc += 0x40;
    scoreboard.clear();
    const auto counter = amdgpu::WaitCounterKind::Load;
    scoreboard.add(
        {scoreboard.issue(counter), 0x100, 1, {RegClass::VGPR, 0, 1}, counter, pending_bytes});
    const auto warnings = cu_->memory_wait_diagnostic_count();
    inputs->v0_bytes = 0;
    cu_->run_issue_cycle(1);
    for (unsigned cycle = 0; cycle < 64 && !wf_->wait_counters().empty(); ++cycle)
      cu_->run_issue_cycle(0);
    EXPECT_EQ(inputs->v0_bytes, 0x3);
    EXPECT_EQ(cu_->memory_wait_diagnostic_count() - warnings, unsigned(pending_bytes == 0x3));
  }
}

TEST_F(Rdna4ImageMemoryTest, SparsePackedStoresFillMissingWordsFromRed) {
  bind(kRgba32Uint);
  cu_->write_vgpr(vb_ + 8, 0, 0x11112222);
  cu_->write_vgpr(vb_ + 9, 0, 0x33334444);
  for (uint8_t mask : {5, 6}) {
    issue(rdna4::kImageStorePckVimage, mask);
    const uint32_t red = (mask & 1) ? 0x11112222u : 0;
    EXPECT_EQ(memory_.read32(kBase), red);
    EXPECT_EQ(memory_.read32(kBase + 4), 0x11112222u);
    EXPECT_EQ(memory_.read32(kBase + 8), 0x33334444u);
    EXPECT_EQ(memory_.read32(kBase + 12), red);
  }
}

TEST_F(Rdna4ImageMemoryTest, AtomicAddTfeReservesStatusOnlyForAResult) {
  for (bool returns : {false, true}) {
    for (uint8_t vdata : {uint8_t(8), uint8_t(31)}) {
      SCOPED_TRACE(testing::Message() << "returns=" << returns << " vdata=" << unsigned(vdata));
      bind(20);
      memory_.write32(kBase, 17);
      cu_->write_vgpr(vb_ + vdata, 0, 5);
      if (vdata == 8)
        cu_->write_vgpr(vb_ + 9, 0, 0xdeadbeef);
      auto words = rdna4::build_vimage(
          rdna4::kImageAtomicAddUintVimage,
          {.dim = 1, .dmask = 1, .vdata = vdata, .th = uint8_t(returns), .tfe = 1, .vaddr1 = 1});
      auto inst = Decoder::create(arch())->decode_window(words);
      ASSERT_TRUE(inst.succeeded());
      ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
      amdgpu::GlobalMemPipeline pipe(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipe.issue(inst.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      cu_->l1_vector().flush_all();
      l2_.flush_all();
      const bool suppressed = returns && vdata == 31;
      EXPECT_EQ(memory_.read32(kBase), suppressed ? 17u : 22u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + vdata, 0), returns && !suppressed ? 17u : 5u);
      if (vdata == 8) {
        EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0xdeadbeefu);
      }
    }
  }
}

TEST_F(Rdna4ImageMemoryTest, OutOfBoundsAtomicSeparatesPayloadReadsFromResultOverwrites) {
  ASSERT_NO_FATAL_FAILURE(check_out_of_bounds_atomic_payload());
}

TEST_F(Rdna4ImageMemoryTest, AtomicTfeReportsOnlyTheNonresidentReturningLane) {
  for (uint32_t level : {0u, 1u})
    ASSERT_NO_FATAL_FAILURE(check_atomic_prt_access(/*compare_swap=*/false, level));
}

TEST_F(Rdna4ImageMemoryTest, CompareSwapTfeWritesStatusOverFirstComparisonWordOnANack) {
  for (uint32_t level : {0u, 1u})
    ASSERT_NO_FATAL_FAILURE(check_atomic_prt_access(/*compare_swap=*/true, level));
}

TEST_F(Rdna4ImageMemoryTest, ImageAtomicsSelectTheirIntegerAndFloatOperation) {
  struct Case {
    uint16_t op;
    uint32_t format, old_value, operand, expected;
  };
  using namespace rdna4;
  const Case cases[] = {
      {kImageAtomicSwapVimage, 20, 7, 3, 3},
      {kImageAtomicAddUintVimage, 20, 7, 3, 10},
      {kImageAtomicSubUintVimage, 20, 7, 3, 4},
      {kImageAtomicMinIntVimage, 21, UINT32_MAX, 3, UINT32_MAX},
      {kImageAtomicMinUintVimage, 20, UINT32_MAX, 3, 3},
      {kImageAtomicMaxIntVimage, 21, UINT32_MAX, 3, 3},
      {kImageAtomicMaxUintVimage, 20, UINT32_MAX, 3, UINT32_MAX},
      {kImageAtomicAndVimage, 20, 6, 3, 2},
      {kImageAtomicOrVimage, 20, 6, 3, 7},
      {kImageAtomicXorVimage, 20, 6, 3, 5},
      {kImageAtomicIncUintVimage, 20, 3, 3, 0},
      {kImageAtomicDecUintVimage, 20, 0, 3, 3},
      {kImageAtomicAddFltVimage, 22, 0x3f800000, 0x40000000, 0x40400000},
      {kImageAtomicMinFltVimage, 22, 0x3f800000, 0x40000000, 0x3f800000},
      {kImageAtomicMaxFltVimage, 22, 0x3f800000, 0x40000000, 0x40000000},
      {kImageAtomicPkAddF16Vimage, 20, 0x40003c00, 0x3c003c00, 0x42004000},
      {kImageAtomicPkAddBf16Vimage, 20, 0x40003f80, 0x3f803f80, 0x40404000},
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(c.op);
    bind(c.format);
    memory_.write32(kBase, c.old_value);
    cu_->write_vgpr(vb_ + 8, 0, c.operand);
    issue(c.op, 1, true);
    EXPECT_EQ(memory_.read32(kBase), c.expected);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), c.old_value);
  }
}

TEST_F(Rdna4ImageMemoryTest, CompareSwapReturnsOldRawWordAndPreservesComparisonRegister) {
  bind(kR32Uint);
  for (uint32_t comparison : {41u, 42u}) {
    memory_.write32(kBase, 41);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    cu_->write_vgpr(vb_ + 8, 0, 99);
    cu_->write_vgpr(vb_ + 9, 0, comparison);
    issue(rdna4::kImageAtomicCmpswapVimage, 3, true);
    EXPECT_EQ(memory_.read32(kBase), comparison == 41 ? 99u : 41u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 41u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), comparison);
  }
}

TEST_F(Rdna4ImageMemoryTest, BoundByteAtomicIsInvalidWithInactiveOrOutOfBoundsLanes) {
  bind(5); // R8_UINT is too narrow for an image atomic.
  auto decoder = Decoder::create(arch());
  for (uint32_t x : {0u, 10000u}) {
    write_coordinates(0, {x, 0, 0});
    for (uint64_t exec : {0u, 1u}) {
      wf_->set_exec(exec);
      auto decoded = decoder->decode_window(rdna4::build_vimage(
          rdna4::kImageAtomicAddUintVimage, {.dim = 1, .dmask = 1, .vdata = 8, .vaddr1 = 1}));
      ASSERT_TRUE(decoded.succeeded());
      EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnsupportedOperandValue);
    }
  }
}

class Cdna2ImageAccessTest : public ImageAccessFixture {
protected:
  /// @brief MIMG flags used by the access fixture.
  struct Encoding {
    bool a16 = false;
    bool d16 = false;
    bool da = true;
    bool unorm = true;
  };

  rj_code_arch_t arch() const override { return ROCJITSU_CODE_ARCH_CDNA2; }

  void bind(const Image &image) {
    const auto words = descriptor(image);
    for (uint32_t i = 0; i < words.size(); ++i)
      cu_->write_sgpr(sb_ + i, words[i]);
  }
  bool issue(bool load, uint32_t dmask = 15) { return issue(load, dmask, Encoding{}); }
  /// @brief Execute IMAGE_LOAD/STORE from s[0:7], VADDR v0, VDATA v8.
  /// @returns Whether the instruction reached the memory pipeline.
  bool issue(bool load, uint32_t dmask, Encoding encoding) {
    cdna2::MimgMachineInst m{};
    m.dmask = dmask;
    m.unorm = encoding.unorm;
    m.da = encoding.da;
    m.a16 = encoding.a16;
    m.d16 = encoding.d16;
    m.vaddr = 0;
    m.vdata = 8;
    m.srsrc = 0;
    const auto *raw = reinterpret_cast<const cdna2::MachineInst *>(&m);
    return load ? run<cdna2::ImageLoadMimg>(raw) : run<cdna2::ImageStoreMimg>(raw);
  }
  void issue_raw(uint16_t opcode, uint8_t mask, bool glc = false, bool acc = false,
                 uint8_t vdata = 8) {
    issue_memory(cdna2::build_mimg(
        opcode, {.dmask = mask, .unorm = 1, .glc = glc, .acc = acc, .vaddr = 0, .vdata = vdata}));
  }
  void issue_memory(std::array<uint32_t, 2> words,
                    std::optional<uint64_t> direct_callback_mask = {}) {
    auto decoder = Decoder::create(arch());
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_TRUE(inst->is_memory_op());
    if (direct_callback_mask) {
      ASSERT_NE(inst->execute, nullptr);
      inst->execute(*inst, wf_);
      ASSERT_NE(inst->data(), nullptr);
      EXPECT_EQ(inst->data_as<amdgpu::VectorMemState>()->exec_mask, *direct_callback_mask);
    } else {
      ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    }
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
  }
  template <typename Inst> bool run(const cdna2::MachineInst *raw) {
    auto inst = std::make_unique<Inst>(raw);
    EXPECT_TRUE(inst->is_memory_op());
    inst->execute_impl(*wf_);
    if (!inst->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    EXPECT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    return true;
  }
};

TEST_F(Cdna2ImageAccessTest, RawImageAtomicsUseAccDataAndReturnOnlyWhenRequested) {
  bind(Image{});
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 5);
  cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeefu);
  const auto acc = vb_ + 256 + 8;
  cu_->write_vgpr(acc, 0, 7);
  cu_->write_vgpr(acc + 1, 0, 5);
  ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageAtomicCmpswap, 3, true, true));
  EXPECT_EQ(memory_.read32(kBase), 7u);
  EXPECT_EQ(cu_->read_vgpr(acc, 0), 5u);
  EXPECT_EQ(cu_->read_vgpr(acc + 1, 0), 5u);
  ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageAtomicAdd, 1, false, true));
  EXPECT_EQ(memory_.read32(kBase), 12u);
  EXPECT_EQ(cu_->read_vgpr(acc, 0), 5u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xdeadbeefu);
}

TEST_F(Cdna2ImageAccessTest, OutOfRangeMemoryDestinationsSuppressTheWholeTransaction) {
  bind(Image{});
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (bool acc : {false, true}) {
    SCOPED_TRACE(acc);
    const uint32_t base = vb_ + (acc ? 256u : 0u);
    memory_.write32(kBase, 5);
    cu_->write_vgpr(base + 32, 0, 7);
    ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageAtomicAdd, 1, true, acc, 32));
    EXPECT_EQ(memory_.read32(kBase), 5u);
    EXPECT_EQ(cu_->read_vgpr(base + 32, 0), 7u);

    // Only the second returned register is out of range. The first must
    // also keep its previous value.
    cu_->write_vgpr(base + 31, 0, 11);
    ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageLoad, 3, false, acc, 31));
    EXPECT_EQ(cu_->read_vgpr(base + 31, 0), 11u);
    EXPECT_EQ(cu_->read_vgpr(base + 32, 0), 7u);
    ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageAtomicAdd, 1, true, acc, 31));
    EXPECT_EQ(memory_.read32(kBase), 16u);
    EXPECT_EQ(cu_->read_vgpr(base + 31, 0), 5u);
  }
  cu_->write_vgpr(vb_, 0, kBase);
  cu_->write_vgpr(vb_ + 1, 0, 0);
  cu_->write_vgpr(vb_ + 8, 0, 11);
  // global_atomic_add v32, v[0:1], v8, off glc
  ASSERT_NO_FATAL_FAILURE(issue_memory({0xdd098000, 0x207f0800}, 0));
  EXPECT_EQ(memory_.read32(kBase), 16u);
  EXPECT_EQ(wf_->exec(), 1u);
}

TEST_F(Cdna2ImageAccessTest, ImageRangeCheckUsesReturnedWidthAndAllowsNonReturningSources) {
  bind(Image{});
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  // A 32-bit compare-swap reads two DWORDs and returns one, even when
  // the source-only compare register is outside the allocation.
  ASSERT_NO_FATAL_FAILURE(
      issue_memory(cdna2::build_mimg(cdna2::kImageAtomicCmpswap,
                                     {.dmask = 3, .unorm = 1, .glc = 1, .vdata = 31}),
                   1));
  ASSERT_NO_FATAL_FAILURE(issue_memory(
      cdna2::build_mimg(cdna2::kImageAtomicAdd, {.dmask = 1, .unorm = 1, .glc = 0, .vdata = 32}),
      1));
  // Two D16 results fit in the last VGPR; three need a second VGPR.
  ASSERT_NO_FATAL_FAILURE(issue_memory(
      cdna2::build_mimg(cdna2::kImageLoad, {.dmask = 3, .unorm = 1, .vdata = 31, .d16 = 1}), 1));
  ASSERT_NO_FATAL_FAILURE(issue_memory(
      cdna2::build_mimg(cdna2::kImageLoad, {.dmask = 7, .unorm = 1, .vdata = 31, .d16 = 1}), 0));
  EXPECT_EQ(wf_->exec(), 1u);
}

TEST_F(Cdna2ImageAccessTest, WideImageCompareSwapSeparatesSourcePairAndComparePair) {
  Image image;
  image.data_format = 11; // two DWORDs per pixel.
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (uint32_t comparison_high : {4u, 5u}) {
    memory_.write32(kBase, 3);
    memory_.write32(kBase + 4, 4);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    for (unsigned i = 0; i < 3; ++i)
      cu_->write_vgpr(vb_ + 8 + i, 0, 1 + i);
    cu_->write_vgpr(vb_ + 11, 0, comparison_high);
    ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageAtomicCmpswap, 15, true));
    EXPECT_EQ(memory_.read32(kBase), comparison_high == 4 ? 1u : 3u);
    EXPECT_EQ(memory_.read32(kBase + 4), comparison_high == 4 ? 2u : 4u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 3u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 4u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 0), 3u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 11, 0), comparison_high);
  }
}

TEST_F(Cdna2ImageAccessTest, PackedImagesBypassConversionAndZeroUnselectedDwords) {
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  Image image;
  image.data_format = 1; // R8 UNORM must deliver integer bits.
  image.pitch = 256;     // GFX9 linear resources have a 256-byte row alignment.
  image.num_format = 0;
  bind(image);
  memory_.write32(kBase, 0xaabbcc80);
  ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageLoadMipPckSgn, 1));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xffffff80u);
  ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageLoadPck, 1));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x80u);
  image.data_format = 14; // Four raw DWORDs.
  bind(image);
  cu_->write_vgpr(vb_ + 8, 0, 0x11111111u);
  cu_->write_vgpr(vb_ + 9, 0, 0x33333333u);
  for (unsigned w = 0; w < 4; ++w)
    memory_.write32(kBase + 4 * w, 0xcccccccc);
  l2_.invalidate_all();
  cu_->l1_vector().invalidate_all();
  ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageStoreMipPck, 5));
  EXPECT_EQ(memory_.read32(kBase), 0x11111111u);
  EXPECT_EQ(memory_.read32(kBase + 4), 0u);
  EXPECT_EQ(memory_.read32(kBase + 8), 0x33333333u);
  EXPECT_EQ(memory_.read32(kBase + 12), 0u);
  ASSERT_NO_FATAL_FAILURE(issue_raw(cdna2::kImageLoadMipPck, 5));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x11111111u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x33333333u);
}

TEST_F(Cdna2ImageAccessTest, StoreThenLoadRoundTripsActiveLanesOfAnR32Image) {
  Image image;
  image.width = 64;
  bind(image);
  for (uint32_t lane = 0; lane < 3; ++lane) {
    write_coordinates(lane, {lane * 16, 0, 0});
    cu_->write_vgpr(vb_ + 8, lane, 0x55000000u + lane);
  }
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase), 0x55000000u);
  EXPECT_EQ(memory_.read32(kBase + 16 * 4), 0u);
  EXPECT_EQ(memory_.read32(kBase + 32 * 4), 0x55000002u);

  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), 0x55000000u + lane);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, lane), 0u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 10, lane), 0u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 11, lane), 1u);
  }
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeef);
}

TEST_F(Cdna2ImageAccessTest, DecodedMipLoadAndStoreAccessLevelZero) {
  Image image;
  image.width = 64;
  image.height = 2;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {1, 1, 0});
  constexpr uint64_t kAddress = kBase + (64 + 1) * sizeof(uint32_t);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);

  for (const bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    memory_.write32(kAddress, 0x12345678u);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeefu);
    const auto words =
        cdna2::build_mimg(load ? cdna2::kImageLoadMipMimg : cdna2::kImageStoreMipMimg,
                          {.dmask = 1, .unorm = 1, .da = 1, .vaddr = 0, .vdata = 8, .srsrc = 0});
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    EXPECT_EQ(memory_.read32(kAddress), load ? 0x12345678u : 0xdeadbeefu);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), load ? 0x12345678u : 0xdeadbeefu);
  }
}

TEST_F(Cdna2ImageAccessTest, ZeroSelectorsDoNotUseBufferStoreA8Routing) {
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (uint32_t format : {kDfmt32, kDfmt8888}) {
    SCOPED_TRACE(format);
    Image image;
    image.data_format = format;
    image.selectors = 0;
    bind(image);
    for (uint32_t reg = 0; reg < 4; ++reg)
      cu_->write_vgpr(vb_ + 8 + reg, 0, format == kDfmt32 ? 0x12345678u : 0x11u * (reg + 1));
    memory_.write32(kBase, 0xdeadbeef);
    ASSERT_TRUE(issue(false));
    EXPECT_EQ(memory_.read32(kBase), 0u);
  }
}

TEST_F(Cdna2ImageAccessTest, ArrayLayersFollowRowsOfPitchElements) {
  Image image;
  image.type = amdgpu::kImage2dArray;
  image.width = 3;
  image.height = 2;
  image.depth = 2;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {2, 1, 1});
  cu_->write_vgpr(vb_ + 8, 0, 0x12345678u);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase + ((1 * 2 + 1) * 64 + 2) * 4), 0x12345678u);
}

TEST_F(Cdna2ImageAccessTest, OneDimensionalArraysTakeTheLayerFromTheSecondCoordinate) {
  Image image;
  image.type = amdgpu::kImage1dArray;
  image.width = 4;
  image.depth = 3;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {1, 2, 0});
  cu_->write_vgpr(vb_ + 8, 0, 0xabcdu);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase + (2 * 64 + 1) * 4), 0xabcdu);
}

TEST_F(Cdna2ImageAccessTest, ArrayWithoutDaUsesLayerZeroAndIgnoresTheNextRegister) {
  wf_->set_exec(1);
  for (uint32_t type : {amdgpu::kImage1dArray, amdgpu::kImage2dArray}) {
    SCOPED_TRACE(type);
    Image image;
    image.type = type;
    image.width = 3;
    image.height = type == amdgpu::kImage1dArray ? 1 : 2;
    image.depth = 2;
    bind(image);
    const uint32_t y = image.height - 1;
    const uint64_t address = kBase + (y * image.pitch + 2) * 4;
    const uint64_t layer_one = address + image.height * image.pitch * 4;
    for (uint32_t unrelated : {1u, 0xffffffffu}) {
      SCOPED_TRACE(unrelated);
      write_coordinates(0, type == amdgpu::kImage1dArray
                               ? std::array<uint32_t, 3>{2, unrelated, unrelated}
                               : std::array<uint32_t, 3>{2, y, unrelated});
      memory_.write32(address, 0x12345678);
      memory_.write32(layer_one, 0x87654321);
      ASSERT_TRUE(issue(true, 1, {.da = false}));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
      cu_->write_vgpr(vb_ + 8, 0, 0x10203040);
      ASSERT_TRUE(issue(false, 1, {.da = false}));
      EXPECT_EQ(memory_.read32(address), 0x10203040u);
      EXPECT_EQ(memory_.read32(layer_one), 0x87654321u);
    }
  }
}

TEST_F(Cdna2ImageAccessTest, A16PacksThreeCoordinatesAndIgnoresTheFinalHighHalf) {
  Image image;
  image.type = amdgpu::kImage2dArray;
  image.width = 3;
  image.height = 2;
  image.depth = 2;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {2 | (1u << 16), 1 | (0xffffu << 16), 0xffffffffu});
  cu_->write_vgpr(vb_ + 8, 0, 0x12345678);
  ASSERT_TRUE(issue(false, 1, {.a16 = true}));
  const uint64_t address = kBase + ((1 * 2 + 1) * image.pitch + 2) * 4;
  EXPECT_EQ(memory_.read32(address), 0x12345678u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 1, {.a16 = true}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0xdeadbeefu);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeefu);
}

TEST_F(Cdna2ImageAccessTest, D16RoundTripsThreeComponentsAndPreservesTheUnusedHighHalf) {
  Image image;
  image.data_format = kDfmt8888;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  cu_->write_vgpr(vb_ + 8, 0, 0x00220011);
  cu_->write_vgpr(vb_ + 9, 0, 0xdead0033);
  ASSERT_TRUE(issue(false, 7, {.d16 = true}));
  EXPECT_EQ(memory_.read32(kBase), 0x00332211u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 7, {.d16 = true}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x00220011u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), cu_->sram_ecc() ? 0x0033u : 0xdead0033u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 0), 0xdeadbeefu);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeefu);
}

TEST_F(Cdna2ImageAccessTest, D16StoresAndLoadsSelectComponentsFromPackedRegisters) {
  Image image;
  image.data_format = kDfmt8888;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  // Preserve sparse component selection while consuming consecutive halves.
  // This follows the explicit RDNA2 Table 41 example; gfx90a is not hardware-tested.
  for (const auto &[mask, expected] :
       {std::pair{0b0011u, 0x00002211u}, std::pair{0b1001u, 0x22000011u},
        std::pair{0b1100u, 0x22110000u}}) {
    SCOPED_TRACE(mask);
    memory_.write32(kBase, 0xffffffff);
    cu_->write_vgpr(vb_ + 8, 0, 0x00220011);
    ASSERT_TRUE(issue(false, mask, {.d16 = true}));
    EXPECT_EQ(memory_.read32(kBase), expected);
  }
  memory_.write32(kBase, 0x44332211);
  ASSERT_TRUE(issue(true, 0b1001, {.d16 = true}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x00440011u);
}

TEST_F(Cdna2ImageAccessTest, XyzOneStoresRetainCompactedSelectorBehavior) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.selectors = kXyzOne;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});

  struct Case {
    uint8_t dmask;
    std::array<uint32_t, 4> packed;
    uint32_t expected;
  };
  const Case cases[] = {
      {15, {0x11, 0x22, 0x33, 0x44}, 0x01332211},
      {7, {0x11, 0x22, 0x33}, 0x00332211},
      {9, {0x11, 0x44}, 0x00000011},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(unsigned{test.dmask});
    memory_.write32(kBase, 0xdeadbeef);
    for (uint32_t component = 0; component < test.packed.size(); ++component)
      cu_->write_vgpr(vb_ + 8 + component, 0, test.packed[component]);
    ASSERT_TRUE(issue(false, test.dmask));
    EXPECT_EQ(memory_.read32(kBase), test.expected);
  }
}

TEST_F(Cdna2ImageAccessTest, ArgbStoreRetainsCompactedSelectorBehavior) {
  constexpr uint32_t kYzwx = 5 | (6 << 3) | (7 << 6) | (4 << 9);
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.selectors = kYzwx;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (uint32_t component = 0; component < 4; ++component)
    cu_->write_vgpr(vb_ + 8 + component, 0, 0x11 * (component + 1));

  ASSERT_TRUE(issue(false, 15));
  EXPECT_EQ(memory_.read32(kBase), 0x33221144u);
}

TEST_F(Cdna2ImageAccessTest, LoadWithoutUnormStillUsesIntegerCoordinates) {
  Image image;
  image.width = 4;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {2, 0, 0});
  memory_.write32(kBase + 8, 0x12345678);
  ASSERT_TRUE(issue(true, 1, {.unorm = false}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
}

TEST_F(Cdna2ImageAccessTest, StoreWithoutRequiredUnormIsRejected) {
  bind(Image{});
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  cu_->write_vgpr(vb_ + 8, 0, 0x12345678);
  EXPECT_FALSE(issue(false, 1, {.unorm = false}));
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnsupportedOperandValue);
  EXPECT_EQ(memory_.read32(kBase), 0u);
}

TEST_F(Cdna2ImageAccessTest, DmaskPacksTheEnabledComponentsAfterTheChannelSelects) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUint;
  image.width = 4;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 0x44332211u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 0b1010));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x22u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x44u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 0), 0xdeadbeef);
}

TEST_F(Cdna2ImageAccessTest, UnormStoresConvertAndLoadsReturnFloats) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = kNfmtUnorm;
  image.width = 4;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {1, 0, 0});
  for (uint32_t i = 0; i < 4; ++i)
    cu_->write_vgpr(vb_ + 8 + i, 0, std::bit_cast<uint32_t>(i == 3 ? 1.0f : 0.0f));
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase + 4), 0xff000000u);
  fill_destination(0);
  ASSERT_TRUE(issue(true));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 11, 0), std::bit_cast<uint32_t>(1.0f));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, OutOfBoundsLanesLoadZeroAndDropTheirStores) {
  Image image;
  image.width = 2;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {2, 0, 0});
  memory_.write32(kBase + 2 * 4, 0x77777777u);
  cu_->write_vgpr(vb_ + 8, 0, 0x11111111u);
  ASSERT_TRUE(issue(false));
  EXPECT_EQ(memory_.read32(kBase + 2 * 4), 0x77777777u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, NullDescriptorLoadsZero) {
  for (uint32_t i = 0; i < 8; ++i)
    cu_->write_sgpr(sb_ + i, 0);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 0x99999999u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + reg, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, CompressedDescriptorsStopBeforeImageMemoryAccess) {
  bind(Image{});
  cu_->write_sgpr(sb_ + 6, cu_->read_sgpr(sb_ + 6) | (1u << 21)); // COMPRESSION_ENABLE.
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (bool load : {false, true}) {
    SCOPED_TRACE(load);
    memory_.write32(kBase, 0x12345678u);
    fill_destination(0xdeadbeefu);
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(issue(load, 1));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xdeadbeefu);
  }
}

TEST_F(Cdna2ImageAccessTest, UnsupportedSwizzleModeStopsBeforeMemoryAccess) {
  Image image;
  image.width = 64;
  image.swizzle_mode = 31; // Reserved swizzle mode.
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  EXPECT_FALSE(issue(true));
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_F(Cdna2ImageAccessTest, ThreeDimensionalSparseStoreUsesPaddedSlicePitch) {
  Image image;
  image.type = amdgpu::kImage3d;
  image.width = 3;
  image.height = 2;
  image.depth = 2;
  image.data_format = kDfmt8888;
  image.selectors = kIdentity;
  bind(image);
  wf_->set_exec(1);
  write_coordinates(0, {2, 1, 1});
  const uint64_t target = kBase + ((image.height + 1) * image.pitch + 2) * 4;
  const uint64_t other_slice = kBase + (image.pitch + 2) * 4;
  memory_.write32(target, 0xffffffff);
  memory_.write32(other_slice, 0xabcdef01);
  cu_->write_vgpr(vb_ + 8, 0, 0x11);
  cu_->write_vgpr(vb_ + 9, 0, 0x22);
  ASSERT_TRUE(issue(false, 0b1001, {.da = false}));
  EXPECT_EQ(memory_.read32(target), 0x22000011u);
  EXPECT_EQ(memory_.read32(other_slice), 0xabcdef01u);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true, 0b1001, {.da = false}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x11u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x22u);
}

TEST_F(Cdna2ImageAccessTest, NullFormatIgnoresConstantOneSelector) {
  Image image;
  image.data_format = 0;
  image.num_format = kNfmtUnorm;
  bind(image);
  wf_->set_exec(1);
  fill_destination(0xdeadbeef);
  ASSERT_TRUE(issue(true));
  for (uint32_t reg = 0; reg < 4; ++reg)
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + reg, 0), 0u);
}

TEST_F(Cdna2ImageAccessTest, DecodedLoadTracksOnlyItsDestinationRegisters) {
  wf_->set_exec(1);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  unsigned reports = 0;
  for (bool d16 : {false, true}) {
    for (uint32_t mask : {0u, 1u, 7u, 9u, 15u}) {
      SCOPED_TRACE(testing::Message() << "d16=" << d16 << " mask=" << mask);
      state.clear();
      reports = 0;
      const auto encoding = cdna2::build_mimg(
          cdna2::kImageLoadMimg,
          {.dmask = static_cast<uint8_t>(mask), .unorm = 1, .vdata = 8, .d16 = d16});
      auto decoded = decoder->decode(encoding.data());
      ASSERT_TRUE(decoded.succeeded());
      cu_->track_memory_wait(*decoded.value(), *wf_);
      state.bind(0x200, &reports,
                 [](void *context, const auto &) { ++*static_cast<unsigned *>(context); });
      const uint16_t words = (std::max(1, std::popcount(mask)) + d16) >> d16;
      state.access({RegClass::VGPR, static_cast<uint16_t>(8 + words), 1}, 1, 0xf, false);
      EXPECT_EQ(reports, 0u);
      state.access({RegClass::VGPR, static_cast<uint16_t>(7 + words), 1}, 1, 0xf, false);
      EXPECT_EQ(reports, 1u);
    }
  }
}

TEST_F(Cdna2ImageAccessTest, DecodedImageChecksTheScaledResourceRegisterRange) {
  wf_->set_exec(1);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  unsigned reports = 0;
  state.bind(0x200, &reports,
             [](void *context, const auto &) { ++*static_cast<unsigned *>(context); });
  for (auto opcode : {cdna2::kImageLoadMimg, cdna2::kImageStoreMimg, cdna2::kImageGetResinfoMimg}) {
    const auto encoding =
        cdna2::build_mimg(opcode, {.dmask = 1, .unorm = 1, .vdata = 8, .srsrc = 4});
    auto decoded = decoder->decode(encoding.data());
    ASSERT_TRUE(decoded.succeeded());
    for (uint16_t reg : {uint16_t{4}, uint16_t{16}}) {
      state.clear();
      reports = 0;
      const auto counter = amdgpu::WaitCounterKind::Ds;
      state.add({state.issue(counter), 0x100, 1, {RegClass::SGPR, reg, 8}, counter, 0xf});
      state.check_instruction(*decoded.value(), *wf_);
      EXPECT_EQ(reports, reg == 16 ? 1u : 0u);
    }
  }
}

TEST_F(Cdna2ImageAccessTest, ImageOnly565FormatStopsAsUnimplemented) {
  Image image;
  image.data_format = 0x10; // GFX9 FMT_5_6_5, outside the buffer DFMT table.
  image.num_format = kNfmtUnorm;
  bind(image);
  EXPECT_FALSE(issue(true));
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_F(Cdna2ImageAccessTest, SrgbLoadDecodesColorAndPreservesAlpha) {
  Image image;
  image.data_format = kDfmt8888;
  image.num_format = 9; // GFX9 TYPE_SRGB.
  image.selectors = kIdentity;
  bind(image);
  memory_.write32(kBase, 0x80402000);
  ASSERT_TRUE(issue(true));
  for (uint32_t lane : {0u, 2u}) {
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), 0u);
    EXPECT_NEAR(std::bit_cast<float>(cu_->read_vgpr(vb_ + 9, lane)), 0.014443844f, 1e-7);
    EXPECT_NEAR(std::bit_cast<float>(cu_->read_vgpr(vb_ + 10, lane)), 0.051269458f, 1e-7);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 11, lane)), 128.0f / 255);
  }
}

TEST(ImageAccessTest, InstructionDimCoordinatesShareA16PackingContract) {
  struct Case {
    uint8_t dim;
    uint32_t coordinates;
  };
  for (const Case test : {
           Case{amdgpu::kImageDim1d, 1},
           Case{amdgpu::kImageDim2d, 2},
           Case{amdgpu::kImageDim3d, 3},
           Case{amdgpu::kImageDimCube, 3},
           Case{amdgpu::kImageDim1dArray, 2},
           Case{amdgpu::kImageDim2dArray, 3},
           Case{amdgpu::kImageDim2dMsaa, 3},
           Case{amdgpu::kImageDim2dMsaaArray, 4},
       }) {
    EXPECT_EQ(amdgpu::image_coordinate_count(test.dim), test.coordinates);
    EXPECT_EQ(amdgpu::image_coordinate_register_count(test.dim, false), test.coordinates);
    EXPECT_EQ(amdgpu::image_coordinate_register_count(test.dim, true), (test.coordinates + 1) / 2);
  }
}

class RdnaDecodedImageAccessTest : public testing::TestWithParam<rj_code_arch_t> {
protected:
  struct Encoding {
    bool r128 = false;
    uint8_t dim = amdgpu::kImageDim2d;
    bool a16 = false;
    bool nsa = true;
    uint8_t dmask = 1;
    bool d16 = false;
    bool tfe = false;
  };

  static constexpr uint32_t kResourceBase = 8;
  amdgpu::GpuMemory memory_{"rdna_image_memory"};
  amdgpu::L2Cache l2_{"rdna_image_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wf_ = nullptr;

  void SetUp() override {
    l2_.set_backing_memory(&memory_);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam();
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 32;
    cfg.vgprs_per_wf = 32;
    ASSERT_NO_FATAL_FAILURE(
        rocjitsu::test::initialize_compute_unit(cu_, "rdna_image_cu", cfg, &memory_, &l2_));
    wf_ = cu_->dispatch_wf(0, 0, 32, 32);
    ASSERT_NE(wf_, nullptr);
    wf_->set_exec(5); // Lane 1 must not store or receive a load result.
    decoder_ = Decoder::create(GetParam());
    ASSERT_NE(decoder_, nullptr);

    bind_resource();
  }

  void bind_resource(const RdnaImageDescriptor &descriptor) {
    const auto resource = rdna_image_descriptor(GetParam(), descriptor);
    for (uint32_t i = 0; i < resource.size(); ++i)
      cu_->write_sgpr(wf_->sgpr_alloc().base + kResourceBase + i, resource[i]);
  }

  void bind_resource(uint32_t format = kR32Uint, uint32_t selectors = kXOne) {
    bind_resource(RdnaImageDescriptor{.format = format, .selectors = selectors});
  }

  void TearDown() override {
    if (wf_)
      wf_->halt();
  }

  DecodeResult decode(bool load) { return decode(load, Encoding{}); }

  DecodeResult decode(bool load, Encoding encoding) {
    const test::Gfx11MimgEncoding mimg{.nsa = encoding.nsa,
                                       .dim = encoding.dim,
                                       .unorm = true,
                                       .dmask = encoding.dmask,
                                       .r128 = encoding.r128,
                                       .a16 = encoding.a16,
                                       .d16 = encoding.d16,
                                       .vaddr = 2,
                                       .vdata = 8,
                                       .srsrc = 2,
                                       .tfe = encoding.tfe,
                                       .nsa_word = 6};
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3) {
      return test::decode_gfx11_mimg<rdna3::MimgBuilderFields>(
          *decoder_, rdna3::build_mimg, load ? rdna3::kImageLoadMimg : rdna3::kImageStoreMimg,
          mimg);
    }
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
      return test::decode_gfx11_mimg<rdna3_5::MimgBuilderFields>(
          *decoder_, rdna3_5::build_mimg, load ? rdna3_5::kImageLoadMimg : rdna3_5::kImageStoreMimg,
          mimg);
    }
    const auto words = rdna4::build_vimage(
        load ? rdna4::kImageLoadVimage : rdna4::kImageStoreVimage, {.dim = encoding.dim,
                                                                    .r128 = encoding.r128,
                                                                    .d16 = encoding.d16,
                                                                    .a16 = encoding.a16,
                                                                    .dmask = encoding.dmask,
                                                                    .vdata = 8,
                                                                    .rsrc = kResourceBase,
                                                                    .tfe = encoding.tfe,
                                                                    .vaddr0 = 2,
                                                                    .vaddr1 = 6});
    return decoder_->decode(words.data());
  }

  DecodeResult decode_mip(bool load) {
    const test::Gfx11MimgEncoding mimg{.nsa = true,
                                       .dim = amdgpu::kImageDim2d,
                                       .unorm = true,
                                       .dmask = 1,
                                       .vaddr = 2,
                                       .vdata = 8,
                                       .srsrc = 2,
                                       .nsa_word = 6};
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3) {
      return test::decode_gfx11_mimg<rdna3::MimgBuilderFields>(
          *decoder_, rdna3::build_mimg, load ? rdna3::kImageLoadMipMimg : rdna3::kImageStoreMipMimg,
          mimg);
    }
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
      return test::decode_gfx11_mimg<rdna3_5::MimgBuilderFields>(
          *decoder_, rdna3_5::build_mimg,
          load ? rdna3_5::kImageLoadMipMimg : rdna3_5::kImageStoreMipMimg, mimg);
    }
    const auto words =
        rdna4::build_vimage(load ? rdna4::kImageLoadMipVimage : rdna4::kImageStoreMipVimage,
                            {.dim = amdgpu::kImageDim2d,
                             .dmask = 1,
                             .vdata = 8,
                             .rsrc = kResourceBase,
                             .vaddr0 = 2,
                             .vaddr1 = 6,
                             .vaddr2 = 7});
    return decoder_->decode(words.data());
  }

  void issue(bool load) { issue(load, Encoding{}); }

  void issue(bool load, Encoding encoding) {
    auto decoded = decode(load, encoding);
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    const InstDefUse def_use(*inst, wf_->wf_size());
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 2, 1}));
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 6, 1}));
    EXPECT_FALSE(def_use.uses.contains(RegisterRef{RegClass::VGPR, 3, 1}));
    EXPECT_TRUE(def_use.uses.contains(RegisterRef{RegClass::SGPR, kResourceBase, 8}));
    EXPECT_FALSE(def_use.uses.contains(RegisterRef{RegClass::SGPR, 2, 1}));

    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
  }
};

TEST_P(RdnaDecodedImageAccessTest, LinearR32DescriptorsPrepareTheExpectedMemoryAccess) {
  wf_->set_exec(1);
  bind_resource(RdnaImageDescriptor{.width = 64, .height = 1, .gfx12_bit_157 = true});
  const uint32_t vb = wf_->vgpr_alloc().base;
  cu_->write_vgpr(vb, 0, 0);
  cu_->write_vgpr(vb + 1, 0, 0);

  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.is_load = true;
  ASSERT_TRUE(amdgpu::prepare_image_access(*wf_, state, kResourceBase,
                                           amdgpu::contiguous_image_address_registers(0),
                                           {.dmask = 15,
                                            .dim = amdgpu::kImageDim2d,
                                            .a16 = false,
                                            .d16 = false,
                                            .lwe = false,
                                            .da = true,
                                            .unorm = true})
                  .succeeded());
  EXPECT_EQ(state.lane_mask, 1u);
  EXPECT_EQ(state.per_lane_addr[0], kBase);
  EXPECT_EQ(state.elem_size, 4u);
  EXPECT_EQ(state.buffer_format, kR32Uint);
}

TEST_P(RdnaDecodedImageAccessTest, DecodedLoadAndStorePrintNoncontiguousCoordinates) {
  for (const bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    auto decoded = decode(load);
    ASSERT_TRUE(decoded.succeeded());
    const std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);
    std::string expected =
        load ? "image_load v8, [v2, v6], s[8:15]" : "image_store v8, [v2, v6], s[8:15]";
    if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3 || GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5)
      expected += " dmask:0x1 dim:SQ_RSRC_IMG_2D unorm";
    EXPECT_EQ(inst->disassemble(), expected);

    const InstDefUse def_use(*inst, wf_->wf_size());
    EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 2, 1}));
    EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 6, 1}));
    EXPECT_FALSE(def_use.uses.contains({RegClass::VGPR, 3, 1}));
  }
}

TEST(ImageAccessTest, Rdna3MipAndAtomicPrintTheirNsaSelectorsAndModifiers) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(decoder, nullptr);
  const test::Gfx11MimgEncoding mimg{.nsa = true,
                                     .dim = amdgpu::kImageDim2d,
                                     .unorm = true,
                                     .dmask = 1,
                                     .vaddr = 2,
                                     .vdata = 8,
                                     .srsrc = 2,
                                     .nsa_word = 6};
  struct Case {
    uint16_t opcode;
    const char *disassembly;
  };
  for (const Case test_case : {
           Case{rdna3::kImageLoadMipMimg,
                "image_load_mip v8, [v2, v6, v0], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm"},
           Case{rdna3::kImageAtomicAddMimg,
                "image_atomic_add v8, [v2, v6], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm"},
       }) {
    SCOPED_TRACE(test_case.disassembly);
    auto decoded = test::decode_gfx11_mimg<rdna3::MimgBuilderFields>(*decoder, rdna3::build_mimg,
                                                                     test_case.opcode, mimg);
    ASSERT_TRUE(decoded.succeeded());
    const std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(), test_case.disassembly);
  }
}

TEST_P(RdnaDecodedImageAccessTest, CompressionControlsStopBeforeImageMemoryAccess) {
  const uint32_t sb = wf_->sgpr_alloc().base + kResourceBase;
  const uint32_t vb = wf_->vgpr_alloc().base;
  const uint32_t original = cu_->read_sgpr(sb + 6);
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  // WRITE_COMPRESS_ENABLE and COMPRESSION_ENABLE, individually and together.
  for (uint32_t flags : {0u, 1u << 20, 1u << 21, 3u << 20}) {
    cu_->write_sgpr(sb + 6, original | flags);
    for (bool load : {false, true}) {
      SCOPED_TRACE(testing::Message() << "flags=" << flags << " load=" << load);
      memory_.write32(kBase, 0x12345678u);
      cu_->write_vgpr(vb + 8, 0, 0xdeadbeefu);
      wf_->clear_instruction_execution_error();
      auto decoded = decode(load);
      ASSERT_TRUE(decoded.succeeded());
      std::unique_ptr<Instruction> inst = std::move(decoded).value();
      EXPECT_EQ(cu_->execute_instruction(inst.get(), *wf_).failed(), flags != 0);
      if (flags == 0) {
        EXPECT_NE(inst->data(), nullptr);
      } else {
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
        EXPECT_EQ(inst->data(), nullptr);
      }
      EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xdeadbeefu);
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, MipLoadAndStoreAccessLevelZero) {
  wf_->set_exec(1);
  const uint32_t vb = wf_->vgpr_alloc().base;
  cu_->write_vgpr(vb + 2, 0, 1);
  cu_->write_vgpr(vb + 6, 0, 1);
  cu_->write_vgpr(vb + 7, 0, 0);
  constexpr uint64_t kAddress = kBase + (64 + 1) * sizeof(uint32_t);

  for (const bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    memory_.write32(kAddress, 0x12345678u);
    cu_->write_vgpr(vb + 8, 0, 0xdeadbeefu);
    auto decoded = decode_mip(load);
    ASSERT_TRUE(decoded.succeeded());
    std::unique_ptr<Instruction> inst = std::move(decoded).value();
    ASSERT_NE(inst, nullptr);

    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    EXPECT_EQ(memory_.read32(kAddress), load ? 0x12345678u : 0xdeadbeefu);
    EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), load ? 0x12345678u : 0xdeadbeefu);
  }
}

TEST_P(RdnaDecodedImageAccessTest,
       StoreThenLoadUsesPaddedRowsNonconsecutiveCoordinatesAndExecMask) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  cu_->write_vgpr(vb + 2, 0, 2);
  cu_->write_vgpr(vb + 6, 0, 1);
  cu_->write_vgpr(vb + 2, 1, 0);
  cu_->write_vgpr(vb + 6, 1, 0);
  cu_->write_vgpr(vb + 2, 2, 1);
  cu_->write_vgpr(vb + 6, 2, 0);
  for (uint32_t lane = 0; lane < 3; ++lane)
    cu_->write_vgpr(vb + 3, lane, 0); // Consecutive-register bug reads this as Y.

  cu_->write_vgpr(vb + 8, 0, 0x11111111);
  cu_->write_vgpr(vb + 8, 1, 0x22222222);
  cu_->write_vgpr(vb + 8, 2, 0x33333333);
  issue(false);
  EXPECT_EQ(memory_.read32(kBase + 264), 0x11111111u);
  EXPECT_EQ(memory_.read32(kBase), 0u);
  EXPECT_EQ(memory_.read32(kBase + 4), 0x33333333u);

  for (uint32_t lane = 0; lane < 3; ++lane)
    cu_->write_vgpr(vb + 8, lane, 0xdeadbeef);
  issue(true);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0x11111111u);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 1), 0xdeadbeefu);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 2), 0x33333333u);
}

TEST_P(RdnaDecodedImageAccessTest, NarrowRowPitchIsRejectedAndUnitWidthUsesQualifiedAlignment) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 1);
  memory_.write32(kBase, 0xaaaaaaaa);
  memory_.write32(kBase + 4, 0xbbbbbbbb);

  bind_resource(RdnaImageDescriptor{.width = 64, .height = 2, .pitch = 1});
  for (bool load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "load=" << load);
    cu_->write_vgpr(vb + 8, 0, 0xdeadbeef);
    auto decoded = decode(load);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(decoded.value()->data(), nullptr);
    EXPECT_EQ(memory_.read32(kBase + 4), 0xbbbbbbbbu);
    EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xdeadbeefu);
  }

  bind_resource(RdnaImageDescriptor{.width = 64, .height = 2, .pitch = 64});
  cu_->write_vgpr(vb + 8, 0, 0x12345678);
  issue(false);
  EXPECT_EQ(memory_.read32(kBase + 64 * 4), 0x12345678u);
  cu_->write_vgpr(vb + 8, 0, 0);
  issue(true);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0x12345678u);

  bind_resource(RdnaImageDescriptor{.width = 1, .height = 2, .pitch = 1});
  cu_->write_vgpr(vb + 8, 0, 0x87654321);
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
    auto decoded = decode(false);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  } else {
    issue(false);
    // Physical gfx1100/gfx1201 captures establish the minimum row quantum.
    const uint32_t row_bytes = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
    EXPECT_EQ(memory_.read32(kBase + row_bytes), 0x87654321u);
  }
  EXPECT_EQ(memory_.read32(kBase + 4), 0xbbbbbbbbu);
}

TEST_P(RdnaDecodedImageAccessTest, StoreDmaskExpandsPackedInputsAndAppliesQualifiedSelectors) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  constexpr uint32_t kBgra = 6 | (5 << 3) | (4 << 6) | (7 << 9);
  struct Case {
    uint32_t selectors;
    uint8_t dmask;
    bool d16;
    std::array<uint32_t, 4> packed;
    std::array<uint32_t, 4> gfx11;
    std::array<uint32_t, 4> gfx12;
  };
  const Case cases[] = {
      {kIdentity, 1, false, {0x11}, {0x11, 0, 0, 0}, {0x11, 0x11, 0x11, 0x11}},
      {kIdentity, 3, false, {0x11, 0x22}, {0x11, 0x22, 0, 0}, {0x11, 0x22, 0x11, 0x11}},
      // GFX12 replicates X only when DMASK supplies X.
      {kIdentity, 2, false, {0x22}, {0, 0x22, 0, 0}, {0, 0x22, 0, 0}},
      // Sparse D16 inputs remain packed in VDATA but route through X/Z.
      {kIdentity, 5, true, {0x00330011}, {0x11, 0, 0x33, 0}, {0x11, 0x11, 0x33, 0x11}},
      {kBgra, 1, false, {0x11}, {0, 0, 0x11, 0}, {0x11, 0x11, 0x11, 0x11}},
      // XYZ1/ZYX1 write constant alpha independently of DMASK.
      {kXyzOne, 15, false, {0x11, 0x22, 0x33, 0x44}, {0x11, 0x22, 0x33, 1}, {0x11, 0x22, 0x33, 1}},
      {kXyzOne, 7, false, {0x11, 0x22, 0x33}, {0x11, 0x22, 0x33, 1}, {0x11, 0x22, 0x33, 1}},
      {kXyzOne, 1, false, {0x11}, {0x11, 0, 0, 1}, {0x11, 0x11, 0x11, 1}},
      {kZyxOne, 15, false, {0x11, 0x22, 0x33, 0x44}, {0x33, 0x22, 0x11, 1}, {0x33, 0x22, 0x11, 1}},
      {kZyxOne, 7, false, {0x11, 0x22, 0x33}, {0x33, 0x22, 0x11, 1}, {0x33, 0x22, 0x11, 1}},
      {kZyxOne, 1, false, {0x11}, {0, 0, 0x11, 1}, {0x11, 0x11, 0x11, 1}},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(testing::Message() << "selectors=" << test.selectors
                                    << " dmask=" << unsigned{test.dmask} << " d16=" << test.d16);
    bind_resource(kRgba32Uint, test.selectors);
    for (uint32_t offset = 0; offset < 4; ++offset) {
      memory_.write32(kBase + offset * 4, 0xdeadbeef);
      cu_->write_vgpr(vb + 8 + offset, 0, test.packed[offset]);
    }
    issue(false, {.dmask = test.dmask, .d16 = test.d16});
    const auto &expected = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? test.gfx12 : test.gfx11;
    for (uint32_t offset = 0; offset < expected.size(); ++offset)
      EXPECT_EQ(memory_.read32(kBase + offset * 4), expected[offset]) << "offset=" << offset;
  }
}

TEST_P(RdnaDecodedImageAccessTest, PackedRgbStoresWriteConstantAlpha) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  constexpr uint32_t kRgba2101010Uint = 40;
  for (const auto &[selectors, expected] :
       {std::pair{kXyzOne, 0x11u | (0x22u << 10) | (0x33u << 20) | (1u << 30)},
        std::pair{kZyxOne, 0x33u | (0x22u << 10) | (0x11u << 20) | (1u << 30)}}) {
    SCOPED_TRACE(selectors);
    bind_resource(kRgba2101010Uint, selectors);
    for (uint32_t component = 0; component < 4; ++component)
      cu_->write_vgpr(vb + 8 + component, 0, 0x11 * (component + 1));
    issue(false, {.dmask = 15});
    EXPECT_EQ(memory_.read32(kBase), expected);
  }
}

TEST_P(RdnaDecodedImageAccessTest, UnmodeledStoreSelectorsReportUnimplemented) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  constexpr uint32_t kXyzz = 4 | (5 << 3) | (6 << 6) | (6 << 9);
  constexpr uint32_t kX000 = 4;
  for (const uint32_t selectors : {kXyzz, kX000}) {
    SCOPED_TRACE(selectors);
    bind_resource(kRgba32Uint, selectors);
    auto decoded = decode(false, {.dmask = 15});
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }

  // The single-channel X001 layout remains supported.
  bind_resource();
  auto decoded = decode(false);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).succeeded());
}

TEST_P(RdnaDecodedImageAccessTest, D16LoadPreservesAndReadsOnlyThePartialDestination) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  memory_.write32(kBase, 0x1234);

  for (const bool d16 : {false, true}) {
    for (const uint8_t dmask : {1, 3, 7, 15}) {
      SCOPED_TRACE(testing::Message() << "d16=" << d16 << " dmask=" << unsigned{dmask});
      for (uint32_t reg = 8; reg < 12; ++reg)
        cu_->write_vgpr(vb + reg, 0, 0xabcdffff);

      auto decoded = decode(true, {.dmask = dmask, .d16 = d16});
      ASSERT_TRUE(decoded.succeeded());
      std::unique_ptr<Instruction> inst = std::move(decoded).value();
      ASSERT_NE(inst, nullptr);

      const uint32_t components = std::popcount(dmask);
      const uint32_t registers = d16 ? (components + 1) / 2 : components;
      const bool preserves_last = d16 && (components & 1u);
      const InstDefUse def_use(*inst, wf_->wf_size());
      EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 8, static_cast<uint8_t>(registers)}));
      for (uint32_t offset = 0; offset < registers; ++offset) {
        const bool expected_use = preserves_last && offset == components / 2;
        EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, static_cast<uint16_t>(8 + offset), 1}),
                  expected_use)
            << "offset=" << offset;
      }

      ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      cu_->l1_vector().flush_all();
      l2_.flush_all();

      for (uint32_t offset = 0; offset < registers; ++offset) {
        uint32_t expected = offset == 0 ? 0x1234 : 0;
        if (!d16 && dmask == 15 && offset == 3)
          expected = 1; // Descriptor W selector is constant one.
        if (d16 && dmask == 15 && offset == 1)
          expected = 1u << 16;
        if (preserves_last && offset == components / 2)
          expected |= 0xabcd0000;
        EXPECT_EQ(cu_->read_vgpr(vb + 8 + offset, 0), expected) << "offset=" << offset;
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, TfeLoadChecksTargetSupportAndPreservesResidentStatus) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  cu_->write_vgpr(vb + 2, 0, 0);
  cu_->write_vgpr(vb + 6, 0, 0);
  cu_->write_vgpr(vb + 8, 0, 0xaaaaaaaa);
  cu_->write_vgpr(vb + 9, 0, 0xdeadbeef);
  memory_.write32(kBase, 0x12345678);

  auto decoded = decode(true, {.tfe = true});
  ASSERT_TRUE(decoded.succeeded());
  std::unique_ptr<Instruction> inst = std::move(decoded).value();
  ASSERT_NE(inst, nullptr);
  const InstDefUse def_use(*inst, wf_->wf_size());
  EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 8, 2}));

  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA1 || GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5 ||
      GetParam() == ROCJITSU_CODE_ARCH_RDNA3) {
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0x12345678u);
    EXPECT_EQ(cu_->read_vgpr(vb + 9, 0), 0xdeadbeefu);
    return;
  }
  EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  EXPECT_EQ(inst->data(), nullptr);
  EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xaaaaaaaau);
  EXPECT_EQ(cu_->read_vgpr(vb + 9, 0), 0xdeadbeefu);
}

TEST_P(RdnaDecodedImageAccessTest, OneDimensionalDimRejectsTwoDimensionalResource) {
  for (const uint8_t dim : {amdgpu::kImageDim1d, amdgpu::kImageDim1dArray}) {
    for (const bool load : {false, true}) {
      for (const bool a16 : {false, true}) {
        for (const bool nsa : {false, true}) {
          if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4 && nsa)
            continue;
          SCOPED_TRACE(testing::Message() << "dim=" << unsigned{dim} << " load=" << load
                                          << " a16=" << a16 << " nsa=" << nsa);
          auto decoded = decode(load, {.dim = dim, .a16 = a16, .nsa = nsa});
          ASSERT_TRUE(decoded.succeeded());
          std::unique_ptr<Instruction> inst = std::move(decoded).value();
          ASSERT_NE(inst, nullptr);
          EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
          EXPECT_EQ(wf_->instruction_execution_error(),
                    amdgpu::InstructionExecutionError::UnimplementedInstruction);
          EXPECT_EQ(inst->data(), nullptr);
        }
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, TwoDimensionalArrayDimUsesTwoDimensionalResourcePrefix) {
  const uint32_t vb = wf_->vgpr_alloc().base;
  wf_->set_exec(1);
  for (const bool a16 : {false, true}) {
    for (const bool nsa : {false, true}) {
      if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4 && nsa)
        continue;
      SCOPED_TRACE(testing::Message() << "a16=" << a16 << " nsa=" << nsa);
      const bool noncontiguous = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 || nsa;
      cu_->write_vgpr(vb + 2, 0, a16 ? 2u | (1u << 16) : 2u);
      cu_->write_vgpr(vb + 3, 0, noncontiguous ? 0xdeadbeef : 1u);
      cu_->write_vgpr(vb + 6, 0, noncontiguous ? 1u : 0xdeadbeef);
      const uint32_t value = 0x12340000u | (uint32_t{a16} << 1) | uint32_t{nsa};
      cu_->write_vgpr(vb + 8, 0, value);

      for (const bool load : {false, true}) {
        auto decoded = decode(load, {.dim = amdgpu::kImageDim2dArray, .a16 = a16, .nsa = nsa});
        ASSERT_TRUE(decoded.succeeded());
        std::unique_ptr<Instruction> inst = std::move(decoded).value();
        ASSERT_NE(inst, nullptr);
        const InstDefUse def_use(*inst, wf_->wf_size());
        EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 2, 1}));
        if (!a16) {
          EXPECT_TRUE(def_use.uses.contains(
              {RegClass::VGPR, static_cast<uint16_t>(noncontiguous ? 6 : 3), 1}));
        }

        ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
        cu_->l1_vector().flush_all();
        l2_.flush_all();
        if (!load) {
          EXPECT_EQ(memory_.read32(kBase + 264), value);
          cu_->write_vgpr(vb + 8, 0, 0xdeadbeef);
        } else {
          EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), value);
        }
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, NonNsaAddressWidthFollowsDimAndA16) {
  if (GetParam() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 VIMAGE encodes address registers individually";

  struct Case {
    uint8_t dim;
    bool a16;
    uint32_t expected_bits;
    const char *address;
  };
  for (const Case test :
       {Case{.dim = amdgpu::kImageDim1d, .a16 = false, .expected_bits = 32, .address = "v2"},
        Case{.dim = amdgpu::kImageDim1d, .a16 = true, .expected_bits = 32, .address = "v2"},
        Case{.dim = amdgpu::kImageDim2d, .a16 = false, .expected_bits = 64, .address = "v[2:3]"},
        Case{.dim = amdgpu::kImageDim2d, .a16 = true, .expected_bits = 32, .address = "v2"}}) {
    for (bool load : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "dim=" << unsigned{test.dim} << " a16=" << test.a16 << " load=" << load);
      auto decoded = decode(load, {.dim = test.dim, .a16 = test.a16, .nsa = false});
      ASSERT_TRUE(decoded.succeeded());
      const std::unique_ptr<Instruction> inst = std::move(decoded).value();
      ASSERT_NE(inst, nullptr);
      const Operand *vaddr = inst->src_operand(load ? 0 : 1);
      ASSERT_NE(vaddr, nullptr);
      EXPECT_EQ(vaddr->size_bits(), test.expected_bits);
      std::string expected =
          std::string(load ? "image_load v8, " : "image_store v8, ") + test.address + ", s[8:15]";
      if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3 || GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
        expected += std::string{" dmask:0x1 dim:SQ_RSRC_IMG_"} +
                    (test.dim == amdgpu::kImageDim1d ? "1D" : "2D") + " unorm";
        if (test.a16)
          expected += " a16";
      }
      EXPECT_EQ(inst->disassemble(), expected);
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, R128LoadAndStoreUseOnlyTheFourDescriptorWordsOnGfx11) {
  const uint32_t sb = wf_->sgpr_alloc().base;
  const uint32_t vb = wf_->vgpr_alloc().base;
  for (uint8_t dim : {amdgpu::kImageDim1d, amdgpu::kImageDim2d}) {
    SCOPED_TRACE(unsigned(dim));
    for (bool poison_trailing_words : {false, true}) {
      SCOPED_TRACE(testing::Message() << "poison_trailing_words=" << poison_trailing_words);
      bind_resource(RdnaImageDescriptor{.type = amdgpu::kImage1d + dim,
                                        .height = dim == amdgpu::kImageDim1d ? 1u : 2u});
      if (poison_trailing_words)
        for (uint32_t reg = kResourceBase + 4; reg < kResourceBase + 8; ++reg)
          cu_->write_sgpr(sb + reg, 0xffffffffu);

      for (bool load : {false, true}) {
        SCOPED_TRACE(testing::Message() << "load=" << load);
        memory_.write32(kBase, 0x12345678u);
        cu_->write_vgpr(vb + 8, 0, 0xdeadbeefu);
        auto decoded = decode(load, {.r128 = true, .dim = dim});
        ASSERT_TRUE(decoded.succeeded());
        std::unique_ptr<Instruction> inst = std::move(decoded).value();
        ASSERT_NE(inst, nullptr);
        if (GetParam() == ROCJITSU_CODE_ARCH_RDNA3 || GetParam() == ROCJITSU_CODE_ARCH_RDNA3_5) {
          wf_->set_exec(1);
          inst->execute(*inst, wf_);
          ASSERT_FALSE(wf_->instruction_execution_failed());
          ASSERT_NE(inst->data(), nullptr);
          amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
          ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
          cu_->l1_vector().flush_all();
          l2_.flush_all();
          EXPECT_EQ(memory_.read32(kBase), load ? 0x12345678u : 0xdeadbeefu);
          EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), load ? 0x12345678u : 0xdeadbeefu);
          continue;
        }
        EXPECT_TRUE(cu_->execute_instruction(inst.get(), *wf_).failed());
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
        EXPECT_EQ(inst->data(), nullptr);
        EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
        EXPECT_EQ(cu_->read_vgpr(vb + 8, 0), 0xdeadbeefu);
      }
    }
  }
}

TEST_P(RdnaDecodedImageAccessTest, LayeredAnd3dLinearImagesUseImplicitRowPitch) {
  wf_->set_exec(1);
  const uint32_t vb = wf_->vgpr_alloc().base;
  for (uint32_t type : {amdgpu::kImage2dArray, amdgpu::kImage3d}) {
    SCOPED_TRACE(testing::Message() << "type=" << type);
    bind_resource(RdnaImageDescriptor{.type = type});
    cu_->write_vgpr(vb, 0, 0);
    cu_->write_vgpr(vb + 1, 0, 1);
    cu_->write_vgpr(vb + 2, 0, 0);

    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.is_load = true;
    EXPECT_TRUE(
        amdgpu::prepare_image_access(
            *wf_, state, kResourceBase, amdgpu::contiguous_image_address_registers(0),
            {.dmask = 1,
             .dim = type == amdgpu::kImage3d ? amdgpu::kImageDim3d : amdgpu::kImageDim2dArray,
             .a16 = false,
             .d16 = false,
             .lwe = false,
             .da = true,
             .unorm = true})
            .succeeded());
    const uint32_t row_bytes = GetParam() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
    EXPECT_EQ(state.per_lane_addr[0], kBase + row_bytes);
    EXPECT_EQ(state.lane_mask, 1u);
  }
}

INSTANTIATE_TEST_SUITE_P(RdnaArchitectures, RdnaDecodedImageAccessTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

TEST(ImageAccessTest, UnsupportedArchitectureIsRejected) {
  amdgpu::GpuMemory memory{"unsupported_image_memory"};
  amdgpu::L2Cache l2{"unsupported_image_l2"};
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 32;
  cfg.vgprs_per_wf = 32;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  ASSERT_NO_FATAL_FAILURE(
      rocjitsu::test::initialize_compute_unit(cu, "unsupported_image_cu", cfg, &memory, &l2));
  auto *wf = cu->dispatch_wf(0, 0, 32, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.is_load = true;
  EXPECT_TRUE(amdgpu::prepare_image_access(
                  *wf, state, 0, amdgpu::contiguous_image_address_registers(0),
                  {.dmask = 1, .a16 = false, .d16 = false, .lwe = false, .da = true, .unorm = true})
                  .failed());
  EXPECT_EQ(wf->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  wf->halt();
}

std::array<uint32_t, 8> rdna_descriptor(const Image &image, rj_code_arch_t arch) {
  std::array<uint32_t, 8> words{};
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const auto set = [&](uint32_t bit, uint32_t count, uint64_t value) {
    for (uint32_t index = 0; index < count; ++index)
      words[(bit + index) / 32] |= ((value >> index) & 1u) << ((bit + index) % 32);
  };
  set(0, 40, kBase >> 8);
  set(gfx12 ? 49 : 52, 8, image.rdna_format);
  set(62, 14, image.width - 1);
  set(78, 14, image.height - 1);
  set(96, 12, kXOne);
  set(124, 4, image.type);
  set(gfx12 ? 57 : 108, gfx12 ? 5 : 4, image.base_level);
  set(gfx12 ? 111 : 112, gfx12 ? 5 : 4, image.last_level);
  const bool max_level_in_word5 =
      arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  set(gfx12 ? 44 : max_level_in_word5 ? 164 : 48, gfx12 ? 5 : 4, image.mip_levels - 1);
  set(max_level_in_word5 ? 40 : gfx12 ? 186 : 187, gfx12 ? 13 : 12, image.min_lod);
  set(180, 3, image.perf_mod);
  set(183, 1, image.corner_samples);
  set(116, 5, image.swizzle_mode);
  if (image.type == amdgpu::kImage3d || image.type == amdgpu::kImage1dArray ||
      image.type == amdgpu::kImage2dArray || image.type == amdgpu::kImageCube ||
      image.type == amdgpu::kImage2dMsaaArray)
    set(128, arch == ROCJITSU_CODE_ARCH_RDNA1 ? 16 : 13, image.depth - 1);
  else if (arch != ROCJITSU_CODE_ARCH_RDNA1)
    set(128, gfx12 ? 16 : 14, image.pitch - 1);
  return words;
}

template <typename Builder, typename Fields>
std::array<uint32_t, 3> image_instruction_words(Builder builder, uint16_t opcode,
                                                const Fields &fields) {
  std::array<uint32_t, 3> words{};
  std::ranges::copy(builder(opcode, fields), words.begin());
  return words;
}

template <typename Builder, typename Fields>
std::array<uint32_t, 3> stub_image_instruction_words(Builder builder, uint16_t opcode,
                                                     Fields fields) {
  fields.srsrc = 4;
  fields.ssamp = 3;
  return image_instruction_words(builder, opcode, fields);
}

TEST(ImageAccessTest, Rdna2ImageResultsMustFitTheLogicalAllocation) {
  amdgpu::GpuMemory memory{"image_memory"};
  amdgpu::L2Cache l2{"image_l2"};
  l2.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA2;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  ASSERT_NO_FATAL_FAILURE(test::initialize_compute_unit(cu, "image_cu", cfg, &memory, &l2));
  auto *wf = cu->dispatch_wf(0, 0, 106, 16, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(15);
  auto decoder = Decoder::create(cfg.arch);
  ASSERT_TRUE(decoder);
  const auto sbase = wf->sgpr_alloc().base;
  const auto vbase = wf->vgpr_alloc().base;
  const auto desc = rdna_descriptor(Image{.rdna_format = 22}, cfg.arch); // R32_FLOAT at kBase.
  for (uint32_t i = 0; i < desc.size(); ++i)
    cu->write_sgpr(sbase + i, desc[i]);
  memory.write32(kBase, 0x3f800000);
  amdgpu::GlobalMemPipeline pipeline(&cu->l1_vector(), &l2);

  struct Case {
    uint16_t opcode;
    uint8_t mask;
    bool d16;
    unsigned words;
    uint32_t first_value;
  };
  constexpr Case cases[] = {
      {rdna2::kImageLoadMimg, 3, false, 2, 0x3f800000},
      {rdna2::kImageLoadMipMimg, 7, true, 2, 0x3c00},
      {rdna2::kImageSampleMimg, 9, false, 2, 0x3f800000},
      {rdna2::kImageSampleDMimg, 3, false, 2, 0x3f800000},
      {rdna2::kImageSampleLMimg, 7, true, 2, 0x3c00},
      {rdna2::kImageSampleLzMimg, 9, false, 2, 0x3f800000},
      {rdna2::kImageGather4LzMimg, 1, false, 4, 0x3f800000},
      {rdna2::kImageGather4LzMimg, 1, true, 2, 0x3c003c00},
      {rdna2::kImageGetResinfoMimg, 3, false, 2, 1},
  };
  constexpr uint32_t sentinel = 0xfeedbeef;
  for (const auto &c : cases) {
    for (unsigned overflow : {0u, 1u}) {
      const auto first = static_cast<uint8_t>(16 - c.words + overflow);
      SCOPED_TRACE(testing::Message()
                   << c.opcode << " d16=" << c.d16 << " first=" << unsigned(first));
      for (uint32_t i = first; i <= 16; ++i)
        cu->write_vgpr(vbase + i, 0, sentinel);
      const bool query = c.opcode == rdna2::kImageGetResinfoMimg;
      auto inst = decoder->decode_window(rdna2::build_mimg(c.opcode, {.dim = amdgpu::kImageDim2d,
                                                                      .dmask = c.mask,
                                                                      .unorm = 1,
                                                                      .tfe = query,
                                                                      .vdata = first,
                                                                      .ssamp = 2,
                                                                      .d16 = c.d16}));
      ASSERT_TRUE(inst.succeeded());
      ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wf).succeeded());
      if (query) {
        EXPECT_EQ(inst.value()->data(), nullptr);
      } else {
        auto *state = inst.value()->data_as<amdgpu::VectorMemState>();
        ASSERT_TRUE(state);
        EXPECT_EQ(state->exec_mask, overflow ? 0u : 15u);
        EXPECT_EQ(state->lane_mask, overflow ? 0u : 15u);
        if (state->image_sample && overflow) {
          for (const auto &tap : state->image_sample->taps)
            EXPECT_EQ(tap.lane_mask, 0u);
        }
        ASSERT_EQ(pipeline.issue(inst.value().release(), *wf), amdgpu::VmAccessOutcome::Complete);
      }
      EXPECT_EQ(wf->exec(), 15u);
      if (overflow) {
        for (uint32_t i = first; i <= 16; ++i)
          EXPECT_EQ(cu->read_vgpr(vbase + i, 0), sentinel);
      } else {
        EXPECT_EQ(cu->read_vgpr(vbase + first, 0), c.first_value);
        EXPECT_EQ(cu->read_vgpr(vbase + 16, 0), sentinel);
      }
    }
  }
  // A suppressed payload still issues an empty transfer with a valid descriptor,
  // including when the result would have an extra status word.
  for (const bool tfe : {false, true}) {
    constexpr uint32_t sentinel = 0xfacefeed;
    cu->write_vgpr(vbase + 15, 0, sentinel);
    auto inst = decoder->decode_window(rdna2::build_mimg(
        rdna2::kImageLoadMimg, {.dim = amdgpu::kImageDim2d, .dmask = 3, .tfe = tfe, .vdata = 15}));
    ASSERT_TRUE(inst.succeeded());
    ASSERT_TRUE(cu->execute_instruction(inst.value().get(), *wf).succeeded());
    const auto *state = inst.value()->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->lane_mask, 0u);
    ASSERT_EQ(pipeline.issue(inst.value().release(), *wf), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(cu->read_vgpr(vbase + 15, 0), sentinel);
  }
  // GFX10.3 validates the descriptor before issuing even an empty image transfer.
  cu->write_sgpr(sbase + 6, desc[6] | (1u << 21));
  for (const uint8_t dst : {8u, 15u}) {
    auto compressed = decoder->decode_window(rdna2::build_mimg(
        rdna2::kImageLoadMimg, {.dim = amdgpu::kImageDim2d, .dmask = 3, .vdata = dst}));
    ASSERT_TRUE(compressed.succeeded());
    EXPECT_TRUE(cu->execute_instruction(compressed.value().get(), *wf).failed());
    EXPECT_EQ(wf->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    wf->clear_instruction_execution_error();
  }
  cu->write_sgpr(sbase + 6, desc[6]);

  // A store has no returned VGPRs; its last allocated source remains usable.
  wf->set_exec(1);
  cu->write_vgpr(vbase + 15, 0, 0x40000000);
  auto store = decoder->decode_window(rdna2::build_mimg(
      rdna2::kImageStoreMimg, {.dim = amdgpu::kImageDim2d, .dmask = 1, .unorm = 1, .vdata = 15}));
  ASSERT_TRUE(store.succeeded());
  ASSERT_TRUE(cu->execute_instruction(store.value().get(), *wf).succeeded());
  ASSERT_EQ(pipeline.issue(store.value().release(), *wf), amdgpu::VmAccessOutcome::Complete);
  EXPECT_EQ(memory.read32(kBase), 0x40000000u);
  wf->halt();
}

class AllImageAccessTest : public ImageAccessFixture,
                           public testing::WithParamInterface<rj_code_arch_t> {
protected:
  bool scattered_ = false;
  bool packed_coordinates_ = false;
  bool packed_data_ = false;
  uint32_t resource_selector_ = 0;
  uint32_t component_mask_ = 1;
  bool unnormalized_ = true;
  bool texel_fault_ = false;
  bool declare_array_ = true;
  std::optional<uint32_t> dimension_override_;
  rj_code_arch_t arch() const override { return GetParam(); }

  void bind_image(uint32_t dimensions, uint32_t width = 3, uint32_t height = 2,
                  uint32_t swizzle = 0) {
    Image image;
    image.width = width;
    image.height = dimensions >= 2 ? height : 1;
    image.swizzle_mode = swizzle;
    image.depth = dimensions == 3 ? 2 : 1;
    image.type = dimensions == 1   ? amdgpu::kImage1d
                 : dimensions == 2 ? amdgpu::kImage2d
                                   : amdgpu::kImage3d;
    bind_image(image);
  }
  void bind_image(const Image &image) {
    const std::array<uint32_t, 8> words =
        arch_is_cdna_4_or_lower(arch()) ? descriptor(image) : rdna_descriptor(image, arch());
    for (uint32_t index = 0; index < words.size(); ++index)
      cu_->write_sgpr(sb_ + index, words[index]);
  }

  template <typename Fields> Fields fields(uint32_t dimensions) {
    Fields encoding{};
    encoding.dmask = component_mask_;
    if constexpr (requires { encoding.tfe; })
      encoding.tfe = texel_fault_;
    encoding.a16 = packed_coordinates_;
    encoding.d16 = packed_data_;
    if constexpr (requires { encoding.srsrc; })
      encoding.srsrc = resource_selector_;
    else
      encoding.rsrc = resource_selector_ * 4;
    encoding.vdata = 8;
    if constexpr (requires { encoding.vaddr; })
      encoding.vaddr = 0;
    else {
      encoding.vaddr0 = 0;
      encoding.vaddr1 = scattered_ ? 10 : 1;
      encoding.vaddr2 = scattered_ ? 12 : 2;
    }
    if constexpr (requires { encoding.nsa; })
      encoding.nsa = scattered_;
    if constexpr (requires { encoding.unorm; })
      encoding.unorm = unnormalized_;
    if constexpr (requires { encoding.da; })
      encoding.da = declare_array_;
    if constexpr (requires { encoding.dim; })
      encoding.dim = dimension_override_.value_or(dimensions - 1);
    return encoding;
  }
  template <typename Fields, typename Builder>
  std::array<uint32_t, 3> sample_mimg_words(Builder builder, uint16_t opcode, bool d16,
                                            uint32_t sampler) {
    Fields encoding = fields<Fields>(2);
    encoding.unorm = 0;
    encoding.ssamp = sampler / 4;
    encoding.d16 = d16;
    return image_instruction_words(builder, opcode, encoding);
  }

  std::unique_ptr<Instruction> decoded_instruction(bool load, uint32_t dimensions,
                                                   bool explicit_mip = false) {
    std::array<uint32_t, 3> words{};
    switch (arch()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      words = image_instruction_words(
          cdna1::build_mimg,
          load ? (explicit_mip ? cdna1::kImageLoadMipMimg : cdna1::kImageLoadMimg)
               : (explicit_mip ? cdna1::kImageStoreMipMimg : cdna1::kImageStoreMimg),
          fields<cdna1::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_CDNA2:
      words = image_instruction_words(
          cdna2::build_mimg,
          load ? (explicit_mip ? cdna2::kImageLoadMipMimg : cdna2::kImageLoadMimg)
               : (explicit_mip ? cdna2::kImageStoreMipMimg : cdna2::kImageStoreMimg),
          fields<cdna2::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA1:
      words = image_instruction_words(
          rdna1::build_mimg,
          load ? (explicit_mip ? rdna1::kImageLoadMipMimg : rdna1::kImageLoadMimg)
               : (explicit_mip ? rdna1::kImageStoreMipMimg : rdna1::kImageStoreMimg),
          fields<rdna1::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA2:
      words = image_instruction_words(
          rdna2::build_mimg,
          load ? (explicit_mip ? rdna2::kImageLoadMipMimg : rdna2::kImageLoadMimg)
               : (explicit_mip ? rdna2::kImageStoreMipMimg : rdna2::kImageStoreMimg),
          fields<rdna2::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      words = image_instruction_words(
          rdna3::build_mimg,
          load ? (explicit_mip ? rdna3::kImageLoadMipMimg : rdna3::kImageLoadMimg)
               : (explicit_mip ? rdna3::kImageStoreMipMimg : rdna3::kImageStoreMimg),
          fields<rdna3::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      words = image_instruction_words(
          rdna3_5::build_mimg,
          load ? (explicit_mip ? rdna3_5::kImageLoadMipMimg : rdna3_5::kImageLoadMimg)
               : (explicit_mip ? rdna3_5::kImageStoreMipMimg : rdna3_5::kImageStoreMimg),
          fields<rdna3_5::MimgBuilderFields>(dimensions));
      break;
    case ROCJITSU_CODE_ARCH_RDNA4:
      words = image_instruction_words(
          rdna4::build_vimage,
          load ? (explicit_mip ? rdna4::kImageLoadMipVimage : rdna4::kImageLoadVimage)
               : (explicit_mip ? rdna4::kImageStoreMipVimage : rdna4::kImageStoreVimage),
          fields<rdna4::VimageBuilderFields>(dimensions));
      break;
    default:
      return {};
    }
    if (scattered_ && arch() != ROCJITSU_CODE_ARCH_RDNA4)
      words[2] = 10 | (12 << 8);
    std::unique_ptr<Decoder> decoder = Decoder::create(arch());
    DecodeResult decoded = decoder->decode(words.data());
    EXPECT_TRUE(decoded.succeeded());
    if (decoded.failed())
      return {};
    return std::move(decoded).value();
  }
  bool decoded_issue(bool load, uint32_t dimensions, bool mip = false) {
    std::unique_ptr<Instruction> inst = decoded_instruction(load, dimensions, mip);
    if (!inst)
      return false;
    EXPECT_TRUE(inst->is_memory_op());
    inst->execute(*inst, wf_);
    if (!inst->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    EXPECT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    return true;
  }
};

class ImageQueryFootprintTest : public AllImageAccessTest {
protected:
  void check_resource_query_inputs() {
    class QueryReads final : public ExecutionPlugin {
    public:
      QueryReads() : ExecutionPlugin("query_reads") {}
      std::array<uint64_t, 2> coords{};
      uint16_t scalars = 0;
      void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                                 uint8_t) override {
        const auto index = reg - wave->vgpr_alloc().base;
        if (index < coords.size())
          coords[index] |= lanes;
      }
      void onAmdgpuReadSgpr(const amdgpu::Wavefront *wave, uint32_t reg) override {
        const auto index = reg - wave->sgpr_alloc().base;
        if (index < 12)
          scalars |= 1u << index;
      }
    };
    bind_image(2, 4, 4);
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto observer = std::make_unique<QueryReads>();
    auto *reads = observer.get();
    ASSERT_TRUE(group->add(std::move(observer)));
    cu_->set_plugin_group(group);
    wf_->set_exec(1);
    auto decoder = Decoder::create(arch());
    for (unsigned mask : {0u, 1u, 3u, 8u}) {
      // One channel fits at the final VGPR. A second channel suppresses the whole query.
      const uint8_t dst = mask == 1 || mask == 3 ? wf_->num_vgprs() - 1 : 8;
      const bool reads_inputs = mask && mask != 3;
      auto decode = [&] {
        if (arch() == ROCJITSU_CODE_ARCH_RDNA4) {
          rdna4::VimageBuilderFields fields{};
          fields.dim = 1;
          fields.dmask = mask;
          fields.vdata = dst;
          return decoder->decode_window(rdna4::build_vimage(rdna4::kImageGetResinfoVimage, fields));
        }
        if (arch() == ROCJITSU_CODE_ARCH_RDNA3 || arch() == ROCJITSU_CODE_ARCH_RDNA3_5) {
          rdna3::MimgBuilderFields fields{};
          fields.dim = 1;
          fields.dmask = mask;
          fields.vdata = dst;
          return decoder->decode_window(rdna3::build_mimg(rdna3::kImageGetResinfoMimg, fields));
        }
        rdna1::MimgBuilderFields fields{};
        fields.dim = 1;
        fields.dmask = mask;
        fields.vdata = dst;
        return decoder->decode_window(rdna1::build_mimg(rdna1::kImageGetResinfoMimg, fields));
      };
      auto inst = decode();
      ASSERT_TRUE(inst.succeeded());
      const InstDefUse def_use(*inst.value(), wf_->wf_size());
      const bool inputs = mask != 0;
      EXPECT_EQ(def_use.uses.contains(RegisterRef{RegClass::VGPR, 0, 1}), inputs);
      EXPECT_EQ(def_use.uses.contains(RegisterRef{RegClass::SGPR, 0, 8}), inputs);
      EXPECT_EQ(def_use.uses.contains(RegisterRef{RegClass::SGPR, 8, 4}), false);
      auto &score = wf_->ensure_memory_wait_scoreboard();
      auto pending = [&](RegisterRef reg, unsigned lane) {
        unsigned hazards = 0;
        score.clear();
        score.bind(0x200, &hazards, [](void *data, const auto &hazard) {
          if (!hazard.write)
            ++*static_cast<unsigned *>(data);
        });
        constexpr auto kind = amdgpu::WaitCounterKind::Load;
        score.add({score.issue(kind), 0x100, uint64_t{1} << lane, reg, kind, 0xf});
        score.check_instruction(*inst.value(), *wf_);
        score.clear();
        return hazards != 0;
      };
      reads->coords.fill(0);
      reads->scalars = 0;
      cu_->write_vgpr(vb_ + 8, 0, 0xfeedbeef);
      cu_->write_vgpr(vb_ + 8, 1, 0xfeedbeef);
      cu_->write_vgpr(vb_ + 9, 0, 0xfeedbeef);
      ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
      if (mask == 0) {
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xfeedbeefu);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0xfeedbeefu);
      }
      for (unsigned reg = 0; reg < 2; ++reg)
        for (unsigned lane : {0u, 1u, 2u, 3u}) {
          const bool actual = (reads->coords[reg] >> lane) & 1u;
          EXPECT_EQ(actual, bool(reads_inputs && reg == 0 && lane == 0));
          EXPECT_EQ(pending({RegClass::VGPR, static_cast<uint16_t>(reg), 1}, lane), actual);
        }
      const uint16_t expected_scalars = reads_inputs ? 0xffu : 0u;
      EXPECT_EQ(reads->scalars, expected_scalars);
      for (uint16_t s : {0u, 7u, 8u, 11u})
        EXPECT_EQ(pending({RegClass::SGPR, s, 1}, 0), bool(expected_scalars & (1u << s)));
    }
  }
};
INSTANTIATE_TEST_SUITE_P(RdnaTargets, ImageQueryFootprintTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

TEST_P(ImageQueryFootprintTest, ResourceQueryWaitsFollowTheChannelMask) {
  check_resource_query_inputs();
}

class Gfx10PixelTransferTest : public AllImageAccessTest {
protected:
  bool transfer(uint16_t opcode, uint32_t dmask, bool r128 = false) {
    component_mask_ = dmask;
    auto rdna1_fields = fields<rdna1::MimgBuilderFields>(2);
    auto rdna2_fields = fields<rdna2::MimgBuilderFields>(2);
    rdna1_fields.r128 = rdna2_fields.r128 = r128;
    const auto words = arch() == ROCJITSU_CODE_ARCH_RDNA1 ? rdna1::build_mimg(opcode, rdna1_fields)
                                                          : rdna2::build_mimg(opcode, rdna2_fields);
    auto decoder = Decoder::create(arch());
    auto decoded = decoder->decode(words.data());
    if (decoded.failed())
      return false;
    auto inst = std::move(decoded).value();
    inst->execute(*inst, wf_);
    if (!inst->data()) {
      ADD_FAILURE() << "image issue failed, execution error "
                    << unsigned(wf_->instruction_execution_error());
      return false;
    }
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    const auto outcome = pipeline.issue(inst.release(), *wf_);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    return outcome == amdgpu::VmAccessOutcome::Complete;
  }
};

INSTANTIATE_TEST_SUITE_P(Gfx10, Gfx10PixelTransferTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2));

TEST_P(Gfx10PixelTransferTest, FloatingImageAtomicsRequireSingleDwordTexels) {
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  auto decoder = Decoder::create(arch());
  for (const uint16_t opcode : {rdna2::kImageAtomicFcmpswapMimg, rdna2::kImageAtomicFminMimg,
                                rdna2::kImageAtomicFmaxMimg, rdna2::kImageAtomicAddMimg}) {
    for (const bool wide : {false, true}) {
      SCOPED_TRACE(testing::Message() << "opcode=" << opcode << " wide=" << wide);
      const bool integer = opcode == rdna2::kImageAtomicAddMimg;
      const bool compare = opcode == rdna2::kImageAtomicFcmpswapMimg;
      Image image;
      constexpr uint32_t kGfx10Rg32Uint = 62;
      constexpr uint32_t kGfx10Rg32Float = 64;
      constexpr uint32_t kGfx10R32Float = 22;
      image.rdna_format =
          integer ? (wide ? kGfx10Rg32Uint : kR32Uint) : (wide ? kGfx10Rg32Float : kGfx10R32Float);
      bind_image(image);
      const uint32_t old = integer ? 5u : std::bit_cast<uint32_t>(2.0f);
      const uint32_t operand =
          integer ? 3u
                  : std::bit_cast<uint32_t>(opcode == rdna2::kImageAtomicFmaxMimg ? 3.0f : 1.0f);
      memory_.write32(kBase, old);
      memory_.write32(kBase + 4, 7);
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      cu_->write_vgpr(vb_ + 8, 0, operand);
      cu_->write_vgpr(vb_ + 9, 0, compare ? old : 0u);
      const uint32_t words = (wide ? 2u : 1u) * (compare ? 2u : 1u);
      auto fields = rdna2::MimgBuilderFields{};
      fields.dim = 1;
      fields.unorm = fields.glc = 1;
      fields.dmask = (1u << words) - 1;
      fields.vdata = 8;
      auto decoded = decoder->decode_window(rdna2::build_mimg(opcode, fields));
      ASSERT_TRUE(decoded.succeeded());
      auto inst = std::move(decoded).value();
      const auto issued = cu_->execute_instruction(inst.get(), *wf_);
      if (!integer && wide) {
        EXPECT_TRUE(issued.failed());
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnsupportedOperandValue);
        EXPECT_EQ(inst->data(), nullptr);
        EXPECT_EQ(memory_.read32(kBase), old);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), operand);
        continue;
      }
      ASSERT_TRUE(issued.succeeded());
      amdgpu::GlobalMemPipeline pipe(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipe.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(memory_.read32(kBase), integer ? 8u : operand);
      EXPECT_EQ(memory_.read32(kBase + 4), 7u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), old);
    }
  }
}

TEST_P(AllImageAccessTest, StoresDiscardOnlyExplicitSparseTexels) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA2 && !amdgpu::has_gfx11_plus_image_operands(arch()))
    GTEST_SKIP();
  class Pages final : public amdgpu::AddressSpaceTranslator {
  public:
    amdgpu::IdentityAddressSpaceTranslator identity;
    bool mapped = true;
    bool sparse = false;
    amdgpu::VmTranslationResult translate(uint64_t address, size_t width,
                                          amdgpu::VmAccessKind kind) const override {
      if (!mapped && address >= kBase + 4096 && address < kBase + 8192)
        return {};
      return identity.translate(address, width, kind);
    }
    std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
      static const auto hole = std::make_shared<const amdgpu::PrtRanges>(
          amdgpu::PrtRanges{{kBase + 4096, kBase + 8192}});
      return sparse ? hole : nullptr;
    }
  };
  auto pages = std::make_shared<Pages>();
  amdgpu::GpuVm vm;
  unsigned faults = 0;
  auto handle = vm.register_address_space(
      1, pages, std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_),
      [&](uint64_t, amdgpu::VmAccessKind) { ++faults; });
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  const auto resource =
      rdna_image_descriptor(arch(), {.format = 20, .width = 2048, .height = 1, .pitch = 2048});
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  for (bool packed : {false, true}) {
    const auto gfx10_words =
        rdna2::build_mimg(packed ? rdna2::kImageStorePckMimg : rdna2::kImageStoreMimg,
                          {.dim = 1, .dmask = 1, .unorm = 1, .vdata = 8});
    const auto gfx11_words =
        rdna3_5::build_mimg(packed ? rdna3_5::kImageStorePckMimg : rdna3_5::kImageStoreMimg,
                            {.dim = 1, .unorm = 1, .dmask = 1, .vdata = 8});
    const auto gfx12_words =
        rdna4::build_vimage(packed ? rdna4::kImageStorePckVimage : rdna4::kImageStoreVimage,
                            {.dim = 1, .dmask = 1, .vdata = 8, .vaddr1 = 1});
    const std::span<const uint32_t> words =
        arch() == ROCJITSU_CODE_ARCH_RDNA2   ? std::span<const uint32_t>(gfx10_words)
        : arch() == ROCJITSU_CODE_ARCH_RDNA4 ? std::span<const uint32_t>(gfx12_words)
                                             : std::span<const uint32_t>(gfx11_words);
    auto decoder = Decoder::create(arch());
    wf_->set_exec(3);
    write_coordinates(0, {0, 0, 0});
    write_coordinates(1, {1024, 0, 0});
    for (const bool sparse : {false, true}) {
      for (const bool mapped : {true, false}) {
        SCOPED_TRACE(testing::Message() << "sparse=" << sparse << " mapped=" << mapped);
        pages->mapped = mapped;
        pages->sparse = sparse;
        faults = 0;
        memory_.write32(kBase, 17);
        memory_.write32(kBase + 4096, 31);
        cu_->write_vgpr(vb_ + 8, 0, 22);
        cu_->write_vgpr(vb_ + 8, 1, 36);
        auto inst = decoder->decode_window(words);
        ASSERT_TRUE(inst.succeeded());
        SCOPED_TRACE(inst.value()->disassemble());
        ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
        amdgpu::GlobalMemPipeline pipe(&cu_->l1_vector(), &l2_);
        const auto outcome = pipe.issue(inst.value().release(), *wf_);
        cu_->l1_vector().flush_all();
        l2_.flush_all();
        const auto resident = memory_.read32(kBase);
        const auto hole = memory_.read32(kBase + 4096);
        EXPECT_EQ(outcome, mapped || sparse ? amdgpu::VmAccessOutcome::Complete
                                            : amdgpu::VmAccessOutcome::Faulted);
        EXPECT_EQ(faults, unsigned(!mapped && !sparse));
        if (mapped || sparse) {
          EXPECT_EQ(resident, 22u);
        }
        EXPECT_EQ(hole, mapped && !sparse ? 36u : 31u);
      }
    }
  }
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(AllImageAccessTest, SparseImageRoutingIsIndependentOfDebuggerProbes) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  class MissingPage final : public amdgpu::AddressSpaceTranslator {
    amdgpu::IdentityAddressSpaceTranslator identity_;
    amdgpu::VmTranslationResult translate(uint64_t addr, size_t size,
                                          amdgpu::VmAccessKind access) const override {
      if ((addr >> 12) == (kBase + 4096) >> 12)
        return {};
      return identity_.translate(addr, size, access);
    }
    std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
      static const auto holes = std::make_shared<const amdgpu::PrtRanges>(
          amdgpu::PrtRanges{{kBase + 4096, kBase + 8192}});
      return holes;
    }
  };

  class Observer final : public ExecutionPlugin {
  public:
    Observer() : ExecutionPlugin("sparse-image-observer") {}
    bool observes_memory_routing() const override { return true; }
    void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access) override {
      masks.push_back(access.request_lane_mask);
      for (const auto &tap : access.image_sample_footprints)
        tap_mask |= tap.lane_mask;
    }
    std::vector<uint64_t> masks;
    uint64_t tap_mask = 0;
  };

  amdgpu::GpuVm vm;
  const auto handle =
      vm.register_address_space(1, std::make_shared<MissingPage>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_), {});
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_exec(3);
  memory_.write32(kBase, 0x3f800000);
  memory_.write32(kBase + 4, 0x3f800000);
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  auto resource =
      rdna_image_descriptor(arch(), {.format = 22, .width = 2048, .height = 1, .pitch = 2048});
  resource[5] = gfx12 ? 1u << 5 : 1u << 8;
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  cu_->write_sgpr(sb_ + 8, 2 | (2 << 3));
  if (gfx12)
    cu_->write_sgpr(sb_ + 10, (1u << 20) | (1u << 22));
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Observer>();
  auto *observed = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  uint64_t pc = 0x4000;
  for (bool debug : {false, true}) {
    cu_->set_debug_active(debug);
    unsigned faults = 0;
    cu_->set_memory_violation_handler([&](amdgpu::Wavefront &, uint64_t, bool, uint64_t) {
      ++faults;
      return true;
    });
    for (bool missing : {false, true}) {
      observed->masks.clear();
      observed->tap_mask = 0;
      fill_destination(0xdeadbeef);
      write_coordinates(0, {0, 0, 0});
      const auto x = missing ? 1024u : 0u;
      if (gfx12) {
        write_coordinates(0, {std::bit_cast<uint32_t>(0.75f), std::bit_cast<uint32_t>(0.5f), 0});
        write_coordinates(
            1, {std::bit_cast<uint32_t>(float(x) + 0.75f), std::bit_cast<uint32_t>(0.5f), 0});
        rdna4::VsampleBuilderFields fields{};
        fields.dim = fields.unorm = fields.dmask = fields.tfe = fields.vaddr1 = 1;
        fields.vdata = fields.samp = 8;
        const auto words = rdna4::build_vsample(rdna4::kImageSampleLzVsample, fields);
        for (unsigned i = 0; i < words.size(); ++i)
          memory_.write32(pc + 4 * i, words[i]);
      } else {
        write_coordinates(1, {x, 0, 0});
        const auto words = rdna3_5::build_mimg(
            rdna3_5::kImageLoadMimg, {.dim = 1, .unorm = 1, .dmask = 1, .vdata = 8, .tfe = 1});
        for (unsigned i = 0; i < words.size(); ++i)
          memory_.write32(pc + 4 * i, words[i]);
      }
      wf_->pc = pc;
      cu_->step();
      const auto req = observed->masks.empty() ? 0 : observed->masks.front();
      EXPECT_FALSE(wf_->instruction_execution_failed());
      ASSERT_EQ(observed->masks.size(), 1u);
      EXPECT_EQ(req, 3u);
      EXPECT_EQ(observed->tap_mask, gfx12 ? 3u : 0u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x3f800000u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), missing ? 0u : 0x3f800000u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 1), missing ? 1u : 0xdeadbeefu);
      EXPECT_EQ(faults, 0u);
      pc += amdgpu::InstructionCache::kLineSize;
    }
    cu_->set_memory_violation_handler({});
  }
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(Gfx10PixelTransferTest, ResidentAndSparseLanesWriteIndependentTexelStatus) {
  class SparsePage final : public amdgpu::AddressSpaceTranslator {
    amdgpu::IdentityAddressSpaceTranslator identity_;
    amdgpu::VmTranslationResult translate(uint64_t address, size_t bytes,
                                          amdgpu::VmAccessKind access) const override {
      return identity_.translate(address, bytes, access);
    }
    std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
      static const auto holes = std::make_shared<const amdgpu::PrtRanges>(
          amdgpu::PrtRanges{{kBase + 4096, kBase + 8192}});
      return holes;
    }
  };
  amdgpu::GpuVm vm;
  const auto handle =
      vm.register_address_space(1, std::make_shared<SparsePage>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_));
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  Image image;
  image.width = image.pitch = 2048;
  bind_image(image);
  memory_.write32(kBase, 0x12345678);
  memory_.write32(kBase + 4096, 0xcafed00d);
  write_coordinates(0, {0, 0, 0});
  write_coordinates(1, {1024, 0, 0});
  wf_->set_exec(3);
  for (const bool tfe : {false, true}) {
    SCOPED_TRACE(testing::Message() << "tfe=" << tfe);
    constexpr uint32_t sentinel = 0xdeadbeef;
    for (unsigned lane = 0; lane < 3; ++lane) {
      cu_->write_vgpr(vb_ + 8, lane, sentinel);
      cu_->write_vgpr(vb_ + 9, lane, sentinel);
    }
    auto fields1 = fields<rdna1::MimgBuilderFields>(2);
    auto fields2 = fields<rdna2::MimgBuilderFields>(2);
    fields1.dmask = fields2.dmask = 1;
    fields1.tfe = fields2.tfe = tfe;
    auto words = arch() == ROCJITSU_CODE_ARCH_RDNA1
                     ? rdna1::build_mimg(rdna1::kImageLoadMimg, fields1)
                     : rdna2::build_mimg(rdna2::kImageLoadMimg, fields2);
    auto decoded = Decoder::create(arch())->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    ASSERT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(decoded.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 2), sentinel);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), sentinel);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 1), tfe ? 1u : sentinel);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 2), sentinel);
  }
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(Gfx10PixelTransferTest, ShortResourceAndMsaaOperandsAddressDistinctFragments) {
  Image image;
  image.type = amdgpu::kImage2dMsaa;
  image.width = image.height = 8;
  image.last_level = 2;    // Four samples.
  image.swizzle_mode = 27; // 64KB_R_X.
  bind_image(image);
  dimension_override_ = amdgpu::kImageDim2dMsaa;
  wf_->set_exec(1);
  // Poison all following SGPRs. R128 brings exactly four descriptor words.
  for (uint32_t i = 4; i < 8; ++i)
    cu_->write_sgpr(sb_ + i, 0xffffffffu);
  // Independent Addr2ComputeSurfaceAddrFromCoord results for (3,7), 32bpp.
  const std::array<uint32_t, 4> offsets = arch() == ROCJITSU_CODE_ARCH_RDNA1
                                              ? std::array<uint32_t, 4>{124, 16508, 32892, 49276}
                                              : std::array<uint32_t, 4>{188, 16572, 32956, 49340};
  for (uint32_t fragment = 0; fragment < 4; ++fragment)
    memory_.write32(kBase + offsets[fragment], 0x1000u + fragment);
  for (uint32_t fragment = 0; fragment < 4; ++fragment) {
    write_coordinates(0, {3, 7, fragment});
    ASSERT_TRUE(transfer(rdna1::kImageLoadMimg, 1, true));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x1000u + fragment);
  }
  write_coordinates(0, {3, 7, 2});
  cu_->write_vgpr(vb_ + 8, 0, 0x2a);
  ASSERT_TRUE(transfer(rdna1::kImageStoreMimg, 1, true));
  EXPECT_EQ(memory_.read32(kBase + offsets[1]), 0x1001u);
  EXPECT_EQ(memory_.read32(kBase + offsets[2]), 0x2au);
  EXPECT_EQ(memory_.read32(kBase + offsets[3]), 0x1003u);
  if (arch() == ROCJITSU_CODE_ARCH_RDNA2) {
    write_coordinates(0, {3, 7, 0});
    ASSERT_TRUE(transfer(rdna2::kImageMsaaLoadMimg, 0xe, true));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x1001u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x2au);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 0), 0x1003u);
    bind_image(image);
    auto decoder = Decoder::create(arch());
    auto &wait = wf_->ensure_memory_wait_scoreboard();
    for (uint8_t mask : {1u, 3u, 7u, 15u}) {
      const unsigned count = std::popcount(mask);
      const uint16_t last = 8 + (count - 1) / 2;
      rdna2::MimgBuilderFields fields{};
      fields.dim = 6;
      fields.dmask = mask;
      fields.vdata = 8;
      fields.unorm = fields.d16 = 1;
      auto inst = decoder->decode_window(rdna2::build_mimg(rdna2::kImageMsaaLoadMimg, fields));
      ASSERT_TRUE(inst.succeeded());
      for (uint8_t bytes : {0x3, 0xc}) {
        SCOPED_TRACE(testing::Message()
                     << "dmask=" << unsigned(mask) << " bytes=" << unsigned(bytes));
        unsigned warnings = 0;
        const auto kind = amdgpu::WaitCounterKind::Ds;
        wait.clear();
        wait.bind(0x200, &warnings,
                  [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
        wait.add({wait.issue(kind), 0x100, 1, {RegClass::VGPR, last, 1}, kind, bytes});
        wait.check_instruction(*inst.value(), *wf_);
        EXPECT_EQ(warnings, unsigned(bytes == 0x3 || !(count & 1)));
      }
      wait.clear();
      cu_->write_vgpr(vb_ + last, 0, 0xbeefdead);
      ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
      amdgpu::GlobalMemPipeline pipe(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipe.issue(inst.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(cu_->read_vgpr(vb_ + last, 0) >> 16, count & 1    ? 0xbeefu
                                                     : count == 2 ? 0x1001u
                                                                  : 0x1003u);
    }
  }
}

TEST_P(Gfx10PixelTransferTest, PackedAndFormattedHorizontalLoadsRespectIndividualTexelBounds) {
  Image image;
  image.pitch = 256; // GFX10 linear rows start on a 256-byte boundary.
  image.width = 4;
  image.rdna_format = 1; // R8_UNORM
  bind_image(image);
  wf_->set_exec(1);
  write_coordinates(0, {1, 0, 0});
  memory_.write32(kBase, 0xff408002);
  ASSERT_TRUE(transfer(rdna1::kImageLoadPckSgnMimg, 1));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xffffff80u);
  ASSERT_TRUE(transfer(rdna1::kImageLoadMipPck2Mimg, 3));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x80u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x40u);
  ASSERT_TRUE(transfer(rdna1::kImageLoadBy4Mimg, 15));
  EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 0)), 128.f / 255);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 9, 0)), 64.f / 255);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 10, 0)), 1.f);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 11, 0), 0u);
}

TEST_P(Gfx10PixelTransferTest, FormattedStoresInvertAChannelPermutation) {
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  Image image;
  image.pitch = 256;      // GFX10 linear rows start on a 256-byte boundary.
  image.rdna_format = 60; // RGBA8_UINT
  bind_image(image);
  constexpr unsigned yzwx = 5 | (6 << 3) | (7 << 6) | (4 << 9);
  cu_->write_sgpr(sb_ + 3, (cu_->read_sgpr(sb_ + 3) & ~0xfffu) | yzwx);
  for (unsigned i = 0; i < 4; ++i)
    cu_->write_vgpr(vb_ + 8 + i, 0, i + 1);
  ASSERT_TRUE(transfer(rdna1::kImageStoreMimg, 15));
  EXPECT_EQ(memory_.read32(kBase), 0x03020104u);
}

TEST_P(Gfx10PixelTransferTest, PackedColorAndDepthStencilUseDescriptorChannelWidths) {
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  Image image;
  image.pitch = 256;       // GFX10 linear rows start on a 256-byte boundary.
  image.rdna_format = 133; // R5G6B5_UNORM
  bind_image(image);
  cu_->write_sgpr(sb_ + 3, (cu_->read_sgpr(sb_ + 3) & ~0xfffu) | kIdentity);
  memory_.write32(kBase, 0xf81f);
  ASSERT_TRUE(transfer(rdna1::kImageLoadMimg, 7));
  EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 0)), 1.f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 9, 0)), 0.f);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 10, 0)), 1.f);
  image.rdna_format = 142; // 8_24_UINT, low 24-bit X followed by 8-bit Y.
  bind_image(image);
  cu_->write_sgpr(sb_ + 3, (cu_->read_sgpr(sb_ + 3) & ~0xfffu) | kIdentity);
  memory_.write32(kBase, 0x12abcdef);
  cu_->l1_vector().invalidate_all();
  l2_.invalidate_all();
  ASSERT_TRUE(transfer(rdna1::kImageLoadMimg, 3));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdefu);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x12u);
}

TEST_P(Gfx10PixelTransferTest, ReadsCompressedBlocksWithPerLaneTexelSelectors) {
  Image image;
  image.pitch = 256; // GFX10 linear rows start on a 256-byte boundary.
  image.width = 8;
  image.height = 4;
  wf_->set_exec(5);
  write_coordinates(0, {1, 0, 0}); // second texel in block 0
  write_coordinates(2, {4, 0, 0}); // first texel in block 1
  for (uint32_t format : {169u, 171u, 173u, 175u, 176u, 177u, 178u}) {
    SCOPED_TRACE(format);
    image.rdna_format = format;
    bind_image(image);
    cu_->write_sgpr(sb_ + 3, (cu_->read_sgpr(sb_ + 3) & ~0xfffu) | kIdentity);
    const bool color = format < 175;
    const uint32_t block_bytes = format == 169 || format == 175 || format == 176 ? 8 : 16;
    for (unsigned block = 0; block < 2; ++block) {
      const uint64_t address = kBase + block * block_bytes;
      for (unsigned word = 0; word < block_bytes; word += 4)
        memory_.write32(address + word, 0);
      if (color) {
        const uint64_t rgb = address + (format == 169 ? 0 : 8);
        memory_.write32(rgb, 0x07e0f800);  // endpoint 0 red, endpoint 1 green
        memory_.write32(rgb + 4, 1u << 2); // selector 1 for the second texel
      } else {
        // Scalar end points, followed by selector 1 for texel 1, 0 for texel 0.
        memory_.write32(address, 0x00080000u | (format % 2 ? 0xffu : 0x81u));
      }
    }
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(transfer(rdna1::kImageLoadMimg, 1));
    EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 0)), 0.f);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 2)),
                    color || format % 2 ? 1.f : -1.f);
  }
}

TEST_P(Gfx10PixelTransferTest, RawMaskedWordsAndConvertedStoresUseDifferentTexelAddresses) {
  Image image;
  image.pitch = 256; // GFX10 linear rows start on a 256-byte boundary.
  image.width = 4;
  image.rdna_format = 1;
  bind_image(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 0xa1b2c3d4);
  cu_->write_vgpr(vb_ + 8, 0, 0x12345678);
  cu_->write_vgpr(vb_ + 9, 0, 0x23456789);
  ASSERT_TRUE(transfer(rdna1::kImageStorePck4Mimg, 5));
  EXPECT_EQ(memory_.read32(kBase), 0xa189c378u);
  cu_->write_vgpr(vb_ + 8, 0, std::bit_cast<uint32_t>(1.f));
  cu_->write_vgpr(vb_ + 9, 0, std::bit_cast<uint32_t>(0.f));
  ASSERT_TRUE(transfer(rdna1::kImageStoreBy2Mimg, 3));
  EXPECT_EQ(memory_.read32(kBase), 0xa18900ffu);
}

TEST_P(AllImageAccessTest, ImplicitSampleUsesQuadDerivativesToSelectTheMip) {
  Image image;
  image.width = image.height = 64;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 0;
  image.mip_levels = 4;
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  const auto bind_view = [&](uint32_t base, uint32_t last) {
    image.base_level = base;
    image.last_level = last;
    bind_image(image);
  };
  const std::array<uint32_t, 4> sampler{2 | (2 << 3) | (2 << 6),
                                        gfx12 ? 0x1fffu << 13 : 0xfffu << 12, 1u << 26, 0};
  for (uint32_t i = 0; i < sampler.size(); ++i)
    cu_->write_sgpr(sb_ + 8 + i, sampler[i]);
  std::array<uint32_t, 3> words{};
  switch (arch()) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    words = sample_mimg_words<cdna1::MimgBuilderFields>(cdna1::build_mimg, cdna1::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_CDNA2:
    words = sample_mimg_words<cdna2::MimgBuilderFields>(cdna2::build_mimg, cdna2::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA1:
    words = sample_mimg_words<rdna1::MimgBuilderFields>(rdna1::build_mimg, rdna1::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA2:
    words = sample_mimg_words<rdna2::MimgBuilderFields>(rdna2::build_mimg, rdna2::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA3:
    words = sample_mimg_words<rdna3::MimgBuilderFields>(rdna3::build_mimg, rdna3::kImageSampleMimg,
                                                        false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    words = sample_mimg_words<rdna3_5::MimgBuilderFields>(rdna3_5::build_mimg,
                                                          rdna3_5::kImageSampleMimg, false, 8);
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    words = rdna4::build_vsample(
        rdna4::kImageSampleVsample,
        {.dim = 1, .dmask = 1, .vdata = 8, .rsrc = 0, .samp = 8, .vaddr0 = 0, .vaddr1 = 1});
    break;
  default:
    FAIL();
  }
  struct QuadCase {
    uint32_t base;
    std::array<std::array<uint32_t, 2>, 4> offsets;
    uint32_t expected_level;
  };
  std::vector<QuadCase> cases;
  for (uint32_t base : {0u, 1u}) {
    for (uint32_t step : {0u, 1u, 2u, 4u, 8u}) {
      cases.push_back({base,
                       {{{0, 0}, {step, 0}, {0, step}, {step, step}}},
                       std::max(base, step <= 1 ? 0u : uint32_t(std::countr_zero(step)))});
    }
  }
  // Independent gfx1100/gfx1201 captures: the same coarse derivatives apply to
  // all four lanes. Moving lane 3 cannot affect them; moving only lane 1 does.
  cases.push_back({0, {{{0, 0}, {0, 0}, {0, 0}, {56, 56}}}, 0});
  cases.push_back({0, {{{0, 0}, {8, 0}, {0, 1}, {1, 1}}}, 3});
  for (const auto &test : cases) {
    wf_->set_exec(15);
    SCOPED_TRACE(testing::Message() << test.base << ',' << test.expected_level);
    const uint32_t level = test.expected_level;
    bind_view(level, level);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      const uint32_t edge = (64u >> level) - 1;
      write_coordinates(lane, {std::min(edge, (8 + test.offsets[lane][0]) >> level),
                               std::min(edge, (8 + test.offsets[lane][1]) >> level), 0});
      cu_->write_vgpr(vb_ + 8, lane, 0xa0000000u + level);
    }
    ASSERT_TRUE(decoded_issue(false, 2));
    bind_view(test.base, 3);
    for (uint32_t lane = 0; lane < 4; ++lane) {
      cu_->write_vgpr(vb_, lane, std::bit_cast<uint32_t>((8.5f + test.offsets[lane][0]) / 64));
      cu_->write_vgpr(vb_ + 1, lane, std::bit_cast<uint32_t>((8.5f + test.offsets[lane][1]) / 64));
      cu_->write_vgpr(vb_ + 8, lane, 0xdeadbeef);
    }
    const auto issue_sample = [&]() {
      auto decoder = Decoder::create(arch());
      auto decoded = decoder->decode(words.data());
      EXPECT_TRUE(decoded.succeeded());
      auto inst = std::move(decoded).value();
      inst->execute(*inst, wf_);
      if (!inst->data())
        return false;
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
      EXPECT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      return true;
    };
    ASSERT_TRUE(issue_sample());
    for (uint32_t lane = 0; lane < 4; ++lane)
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), 0xa0000000u + level);
    EXPECT_TRUE(wf_->wait_counters().empty());
    if (test.base == 0 && (level == 0 || level == 2)) {
      for (uint64_t mask : {1u, 7u}) {
        for (unsigned lane = 0; lane < 4; ++lane)
          cu_->write_vgpr(vb_ + 8, lane, 0xdeadbeef);
        wf_->set_exec(mask);
        wf_->clear_instruction_execution_error();
        const bool helpers =
            arch() == ROCJITSU_CODE_ARCH_RDNA3 || arch() == ROCJITSU_CODE_ARCH_RDNA3_5 || gfx12;
        EXPECT_EQ(issue_sample(), helpers);
        EXPECT_EQ(wf_->instruction_execution_error(),
                  helpers ? amdgpu::InstructionExecutionError::None
                          : amdgpu::InstructionExecutionError::UnimplementedInstruction);
        for (unsigned lane = 0; lane < 4; ++lane)
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane),
                    helpers && (mask & (1u << lane)) ? 0xa0000000u + level : 0xdeadbeef);
        wf_->clear_instruction_execution_error();
        bind_view(0, 0);
        EXPECT_TRUE(issue_sample());
        EXPECT_EQ(wf_->instruction_execution_error(), amdgpu::InstructionExecutionError::None);
        bind_view(0, 3);
      }
    }
  }
}

TEST_P(AllImageAccessTest, MipOpcodesUseViewRelativeLevelsForLoadsAndStores) {
  for (bool a16 : {false, true}) {
    for (bool nsa : {false, true}) {
      if (nsa && arch_is_cdna_4_or_lower(arch()))
        continue;
      packed_coordinates_ = a16;
      scattered_ = nsa;
      SCOPED_TRACE(testing::Message() << "a16=" << a16 << " nsa=" << nsa);
      const auto write_mip = [&](uint32_t level) {
        const uint32_t selector = a16 ? (nsa ? 10 : 1) : (nsa ? 12 : 2);
        cu_->write_vgpr(vb_ + selector, 0, level);
      };
      Image image;
      image.width = image.height = 64;
      image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 0;
      image.mip_levels = 2;
      const auto bind_level = [&](uint32_t base, uint32_t last = 1) {
        image.base_level = base;
        image.last_level = last;
        bind_image(image);
      };
      wf_->set_exec(1);
      for (uint32_t reg : {0u, 1u, 2u, 10u, 12u})
        cu_->write_vgpr(vb_ + reg, 0, 0);
      // Seed the two physical levels through independently qualified plain opcodes.
      bind_level(0);
      cu_->write_vgpr(vb_ + 8, 0, 0x11112222);
      ASSERT_TRUE(decoded_issue(false, 2));
      bind_level(1);
      cu_->write_vgpr(vb_ + 8, 0, 0x33334444);
      ASSERT_TRUE(decoded_issue(false, 2));
      cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
      cu_->write_vgpr(vb_ + 8, 1, 0xdeadbeef);
      write_mip(0);
      ASSERT_TRUE(decoded_issue(true, 2, true));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x33334444u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeefu);
      bind_level(0);
      write_mip(1);
      ASSERT_TRUE(decoded_issue(true, 2, true));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x33334444u);
      cu_->write_vgpr(vb_ + 8, 0, 0x77778888);
      ASSERT_TRUE(decoded_issue(false, 2, true));
      bind_level(1);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x77778888u);
      bind_level(0);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x11112222u);
      bind_level(1);
      for (uint32_t relative : {1u, 0xffffffffu}) {
        SCOPED_TRACE(relative);
        write_mip(relative);
        ASSERT_TRUE(decoded_issue(true, 2, true));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
        cu_->write_vgpr(vb_ + 8, 0, 0xaaaaaaaa);
        ASSERT_TRUE(decoded_issue(false, 2, true)); // OOB cannot change either level.
      }
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x77778888u);
      write_mip(0);
      cu_->write_vgpr(vb_ + 8, 0, 0x55556666);
      ASSERT_TRUE(decoded_issue(false, 2, true));
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x55556666u);
      bind_level(0);
      ASSERT_TRUE(decoded_issue(true, 2));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x11112222u);
    }
  }
}

TEST_P(AllImageAccessTest, DecodedLoadAndStoreRoundTripDimensionsAndExec) {
  for (uint32_t dimensions = 1; dimensions <= 3; ++dimensions) {
    SCOPED_TRACE(dimensions);
    for (bool array_dim : {false, true}) {
      if (array_dim && (dimensions != 3 || arch_is_cdna_4_or_lower(arch())))
        continue;
      SCOPED_TRACE(array_dim);
      dimension_override_ = array_dim ? amdgpu::kImageDim2dArray : dimensions - 1;
      bind_image(dimensions);
      for (uint32_t lane = 0; lane < 3; ++lane) {
        write_coordinates(lane, {lane, dimensions >= 2 ? 1u : 0u, dimensions == 3 ? 1u : 0u});
        cu_->write_vgpr(vb_ + 8, lane, 0x12345678u + lane);
      }
      ASSERT_TRUE(decoded_issue(false, dimensions));
      fill_destination(0xdeadbeef);
      ASSERT_TRUE(decoded_issue(true, dimensions));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeef);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 2), 0x1234567au);
    }
  }
}

TEST_P(AllImageAccessTest, ScaledViewsStopBeforeImageMemoryAccess) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "GFX11 and GFX12 expose separate resource scales";
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  // Exercise each depth, height and width scale bit and GFX12 UAV3D.
  for (uint32_t bit_index = gfx12 ? 4 : 8; bit_index <= 19; ++bit_index) {
    SCOPED_TRACE(bit_index);
    bind_image(3, 64, 64);
    const uint32_t linked = gfx12 && bit_index == 4 ? 0u : 1u << 24;
    cu_->write_sgpr(sb_ + 5, linked | (1u << bit_index));
    for (bool load : {false, true}) {
      SCOPED_TRACE(load);
      memory_.write32(kBase, 0x12345678);
      cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
      wf_->clear_instruction_execution_error();
      EXPECT_FALSE(decoded_issue(load, 3));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
    }
  }
}

TEST_P(AllImageAccessTest, TiledAddressesMatchIndependentBackingOffsets) {
  // Direct ROCr AddrLib reference calls establish these R32 offsets in a
  // 64x64 image. The latter coordinates cross 256-byte and 4-KiB tile edges.
  const rj_code_arch_t architecture = arch();
  const bool gfx9 = arch_is_cdna_4_or_lower(architecture);
  const bool gfx10 =
      architecture == ROCJITSU_CODE_ARCH_RDNA1 || architecture == ROCJITSU_CODE_ARCH_RDNA2;
  const uint32_t swizzle = gfx9 ? 22 : architecture == ROCJITSU_CODE_ARCH_RDNA4 ? 1 : 2;
  bind_image(2, 64, 64, swizzle);
  wf_->set_exec(1);
  const std::array<std::array<uint32_t, 3>, 3> cases{{{1, 1, gfx9 || gfx10 ? 20u : 12u},
                                                      {9, 9,
                                                       gfx9    ? 788u
                                                       : gfx10 ? 2324u
                                                               : 2316u},
                                                      {33, 17, gfx9 || gfx10 ? 5140u : 5132u}}};
  for (const auto &[x, y, offset] : cases) {
    SCOPED_TRACE(testing::Message() << x << ',' << y);
    const uint32_t stored = 0x12340000u + offset;
    const uint32_t loaded = 0xabcd0000u + offset;
    write_coordinates(0, {x, y, 0});
    cu_->write_vgpr(vb_ + 8, 0, stored);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offset), stored);
    EXPECT_NE(memory_.read32(kBase + (y * 64 + x) * 4), stored);
    // Seed backing memory independently before loading, so a mutually wrong
    // store and load cannot satisfy the oracle by undoing each other.
    memory_.write32(kBase + offset, loaded);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), loaded);
  }
}

TEST_P(AllImageAccessTest, Rdna2RbPlusMatchesIndependentBackingOffsets) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP() << "Navi21 RB+ qualification";
  // Independent Navi21 (revision 0x28), GB_ADDR_CONFIG=0x444 witnesses.
  constexpr uint32_t cases[][4] = {
      {2, 4, 0, 64},  {2, 33, 17, 17428}, {6, 4, 0, 64},  {6, 33, 17, 5140},
      {10, 4, 0, 64}, {10, 33, 17, 9236}, {22, 4, 0, 64}, {22, 33, 17, 4116},
      {26, 4, 0, 64}, {26, 33, 17, 8212}, {27, 4, 0, 64}, {27, 33, 17, 2836},
  };
  wf_->set_exec(1);
  for (const auto &c : cases) {
    Image image;
    image.width = 256;
    image.height = 128;
    image.swizzle_mode = c[0];
    bind_image(image);
    write_coordinates(0, {c[1], c[2], 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + c[3]);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + c[3]), 0x12340000u + c[3]);
    memory_.write32(kBase + c[3], 0xabcd0000u + c[3]);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + c[3]);
  }
}

TEST_P(AllImageAccessTest, TiledStoresMatchPhysicalGpuBacking) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical backing captures are available for gfx1100 and gfx1201";
  // ROCr opaque R32 37x29 captures on W7900 and R9700: these texel values
  // occur at exactly one DWORD in each raw backing dump. The last coordinate
  // crosses a 4-KiB tile boundary. No emulator-generated address is the oracle.
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  bind_image(2, 37, 29, gfx12 ? 2 : 22);
  wf_->set_exec(1);
  const std::array<std::array<uint32_t, 4>, 3> cases{{{1, 1, 12, 0xb5f93d83},
                                                      {21, 11, 2412, 0x2167abef},
                                                      {33, 27, gfx12 ? 5420u : 4396u, 0x85c90d53}}};
  for (const auto &[x, y, offset, value] : cases) {
    SCOPED_TRACE(testing::Message() << x << ',' << y);
    write_coordinates(0, {x, y, 0});
    cu_->write_vgpr(vb_ + 8, 0, value);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offset), value);
  }
}

TEST_P(AllImageAccessTest, MipViewsUseTheWholeAllocationAndTheirSelectedLevel) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
  Image image;
  image.width = 37;
  image.height = 19;
  image.mip_levels = 4;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22;
  wf_->set_exec(1);
  // Independent reference constants; the GFX11/12 layouts also match physical
  // full-allocation captures after each mip-level import.
  const std::array<uint32_t, 2> offsets = gfx9 ? std::array<uint32_t, 2>{116, 8308}
                                          : arch() == ROCJITSU_CODE_ARCH_RDNA2
                                              ? std::array<uint32_t, 2>{8308, 4212}
                                          : gfx10 ? std::array<uint32_t, 2>{8372, 4276}
                                                  : std::array<uint32_t, 2>{8300, 4204};
  for (uint32_t level = 0; level < offsets.size(); ++level) {
    SCOPED_TRACE(level);
    image.base_level = image.last_level = level;
    bind_image(image);
    write_coordinates(0, {5, 3, 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + level);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offsets[level]), 0x12340000u + level);
    memory_.write32(kBase + offsets[level], 0xabcd0000u + level);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + level);
  }
  write_coordinates(0, {image.width >> 1, 0, 0});
  cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeefu);
  ASSERT_TRUE(decoded_issue(true, 2));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
}

TEST_P(AllImageAccessTest, LinearRowsMatchPhysicalGpuBackingAtExplicitPitches) {
  if (!amdgpu::has_gfx11_plus_image_operands(arch()))
    GTEST_SKIP();
  Image image;
  image.width = 30;
  wf_->set_exec(1);
  // Captures on gfx1100/gfx1201 establish 256/128-byte row alignment.
  // gfx1150 is qualified here for 256-byte aligned pitches only.
  const bool gfx115 = arch() == ROCJITSU_CODE_ARCH_RDNA3_5;
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t sentinel = 0xdeadbeef;
  auto &waits = wf_->ensure_memory_wait_scoreboard();
  unsigned reports = 0;
  waits.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
  constexpr auto counter = amdgpu::WaitCounterKind::Load;
  for (uint32_t dimensions : {1u, 2u}) {
    image.type = dimensions == 1 ? amdgpu::kImage1d : amdgpu::kImage2d;
    image.height = dimensions == 1 ? 1u : 5u;
    for (uint32_t pitch : {1u, 8u, 15u, 16u, 30u, 32u, 35u, 64u, 100u, 128u}) {
      const bool valid = pitch >= image.width && (!gfx115 || pitch % 64 == 0);
      bind_image(image);
      cu_->write_sgpr(sb_ + 4, pitch - 1);
      for (bool warn : {false, true}) {
        cu_->write_sgpr(sb_ + 5, uint32_t(warn) << (gfx12 ? 5 : 8));
        for (bool tfe : {false, true}) {
          // VIMAGE load/store on this branch rejects TFE before reading the descriptor.
          if (gfx12 && tfe)
            continue;
          texel_fault_ = tfe;
          for (uint32_t row : {0u, 1u, 3u}) {
            if (row >= image.height)
              continue;
            SCOPED_TRACE(testing::Message()
                         << "dimensions=" << dimensions << " pitch=" << pitch << " warn=" << warn
                         << " tfe=" << tfe << " row=" << row);
            write_coordinates(0, {5, row, 0});
            const uint32_t alignment = gfx12 ? 32 : 64;
            const uint32_t row_pitch = (pitch + alignment - 1) & ~(alignment - 1);
            const uint32_t offset = (row * row_pitch + 5) * 4;
            memory_.write32(kBase + offset, sentinel);
            for (bool load : {false, true}) {
              cu_->write_vgpr(vb_ + 8, 0, load ? sentinel : 0x12340000u + offset);
              auto instruction = decoded_instruction(load, dimensions);
              ASSERT_NE(instruction, nullptr);
              waits.clear();
              waits.add({waits.issue(counter), 0x100, 1, {RegClass::VGPR, 0, 1}, counter, 0x8});
              reports = 0;
              waits.check_instruction(*instruction, *wf_);
              EXPECT_EQ(reports != 0, valid);
              waits.clear();
              wf_->clear_instruction_execution_error();
              if (valid) {
                ASSERT_TRUE(decoded_issue(load, dimensions));
                EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + offset);
                EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12340000u + offset);
              } else {
                EXPECT_FALSE(decoded_issue(load, dimensions));
                EXPECT_EQ(wf_->instruction_execution_error(),
                          amdgpu::InstructionExecutionError::UnimplementedInstruction);
                EXPECT_EQ(memory_.read32(kBase + offset), sentinel);
                if (load) {
                  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), sentinel);
                }
              }
            }
          }
        }
      }
    }
  }

  // The explicit pitch describes storage at mip zero. BASE_LEVEL changes the
  // visible extent, but must not reassign the allocation pitch to that mip.
  image.width = 64;
  image.height = 8;
  image.pitch = 128;
  image.mip_levels = 2;
  image.last_level = 1;
  for (uint32_t base_level : {0u, 1u}) {
    image.base_level = base_level;
    bind_image(image);
    for (bool warn : {false, true}) {
      cu_->write_sgpr(sb_ + 5, uint32_t(warn) << (gfx12 ? 5 : 8));
      for (bool tfe : {false, true}) {
        if (gfx12 && tfe)
          continue;
        texel_fault_ = tfe;
        for (uint32_t relative : {0u, 1u}) {
          const uint32_t level = base_level + relative;
          SCOPED_TRACE(testing::Message() << "base=" << base_level << " relative=" << relative
                                          << " warn=" << warn << " tfe=" << tfe);
          write_coordinates(0, {5, 2, relative});
          // Mip 1 starts at zero with 64-texel rows. Mip 0 follows its four rows.
          constexpr uint32_t mip1 = (2 * 64 + 5) * 4;
          constexpr uint32_t mip0 = (4 * 64 + 2 * 128 + 5) * 4;
          constexpr uint32_t wrong_row = (2 * 128 + 5) * 4;
          for (auto offset : {mip0, mip1, wrong_row})
            memory_.write32(kBase + offset, sentinel);
          cu_->l1_vector().invalidate_all();
          l2_.invalidate_all();
          constexpr uint32_t value = 0x1234abcd;
          cu_->write_vgpr(vb_ + 8, 0, value);
          ASSERT_TRUE(decoded_issue(false, 2, relative != 0));
          EXPECT_EQ(memory_.read32(kBase + mip0), level == 0 ? value : sentinel);
          EXPECT_EQ(memory_.read32(kBase + mip1), level == 1 ? value : sentinel);
          EXPECT_EQ(memory_.read32(kBase + wrong_row), sentinel);
          cu_->write_vgpr(vb_ + 8, 0, sentinel);
          ASSERT_TRUE(decoded_issue(true, 2, relative != 0));
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), level <= image.last_level ? value : 0u);
        }
      }
    }
  }
}

TEST_P(AllImageAccessTest, TiledPitchFieldMatchesPhysicalGpuBacking) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical tiled pitch captures cover gfx1100 and gfx1201";
  Image image;
  image.width = 300;
  image.height = 70;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 3 : 27;
  const uint32_t offset = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 156316 : 146588;
  wf_->set_exec(1);
  // Full backing dumps are identical for these descriptor pitches on both GPUs.
  for (uint32_t pitch : {1u, 320u, 384u, 512u, 1024u}) {
    bind_image(image);
    cu_->write_sgpr(sb_ + 4, pitch - 1);
    write_coordinates(0, {299, 69, 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + pitch);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + pitch);
  }
}

TEST_P(AllImageAccessTest, LinearArrayAndVolumeRowsMatchPhysicalGpuBacking) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Physical linear captures are available for gfx1100 and gfx1201";
  Image image;
  image.width = 30;
  image.height = 5;
  image.depth = 2;
  wf_->set_exec(1);
  // Independent raw captures: GFX12 rows advance by 128 bytes, while slices
  // advance by 5 * 256 bytes. GFX11 rows and slice allocation use 256 bytes.
  const uint32_t row_pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
  for (uint32_t type : {amdgpu::kImage2dArray, amdgpu::kImage3d}) {
    image.type = type;
    for (uint32_t dim : {type - amdgpu::kImage1d, uint32_t{amdgpu::kImageDim2dArray}}) {
      SCOPED_TRACE(dim);
      dimension_override_ = dim;
      bind_image(image);
      for (const std::array<uint32_t, 3> coordinates :
           {std::array<uint32_t, 3>{5, 1, 0}, std::array<uint32_t, 3>{5, 3, 1}}) {
        const uint32_t offset =
            coordinates[2] * 1280 + coordinates[1] * row_pitch + coordinates[0] * 4;
        write_coordinates(0, coordinates);
        cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + offset);
        ASSERT_TRUE(decoded_issue(false, 3));
        EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + offset);
        memory_.write32(kBase + offset, 0xabcd0000u + offset);
        cu_->l1_vector().invalidate_all();
        l2_.invalidate_all();
        ASSERT_TRUE(decoded_issue(true, 3));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + offset);
      }
    }
  }
}

TEST_P(AllImageAccessTest, ArrayLayerPositionFollowsInstructionDimension) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 uses DA rather than DIM";
  Image image;
  image.width = 30;
  image.depth = 3;
  image.type = amdgpu::kImage1dArray;
  declare_array_ = false; // GFX10+ DIM supplies the coordinate inventory.
  bind_image(image);
  wf_->set_exec(1);
  for (uint32_t dim : {amdgpu::kImageDim1dArray, amdgpu::kImageDim2dArray, amdgpu::kImageDim3d}) {
    dimension_override_ = dim;
    for (uint32_t layer : {1u, 2u}) {
      SCOPED_TRACE(testing::Message() << "dim=" << dim << " layer=" << layer);
      // Native R32 1D-array captures on gfx1100 (W7900) and gfx1201 (R9700)
      // place each layer on a 256-byte boundary.
      const uint32_t offset = layer * 256 + 5 * 4;
      write_coordinates(0, dim == amdgpu::kImageDim1dArray ? std::array<uint32_t, 3>{5, layer, 0}
                                                           : std::array<uint32_t, 3>{5, 0, layer});
      cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + offset);
      ASSERT_TRUE(decoded_issue(false, 3));
      EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + offset);
      memory_.write32(kBase + offset, 0xabcd0000u + offset);
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      ASSERT_TRUE(decoded_issue(true, 3));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + offset);
    }
  }
}

TEST_P(AllImageAccessTest, ArrayResourcesRejectDimensionsWithoutALayerOperand) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 uses DA rather than DIM";
  Image image;
  image.width = 30;
  image.depth = 3;
  image.type = amdgpu::kImage1dArray;
  bind_image(image);
  wf_->set_exec(1);
  write_coordinates(0, {5, 2, 1});
  for (uint32_t dim : {amdgpu::kImageDim1d, amdgpu::kImageDim2d})
    for (bool load : {false, true}) {
      dimension_override_ = dim;
      wf_->clear_instruction_execution_error();
      EXPECT_FALSE(decoded_issue(load, 3));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
}

TEST_P(AllImageAccessTest, StubMimgInstructionsScaleDescriptorSelectors) {
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 uses VIMAGE/VSAMPLE rather than MIMG";
  for (uint32_t operation = 0; operation < 3; ++operation) {
    std::array<uint32_t, 3> words{};
    switch (arch()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      words = stub_image_instruction_words(cdna1::build_mimg,
                                           operation == 0   ? cdna1::kImageSampleMimg
                                           : operation == 1 ? cdna1::kImageLoadMipMimg
                                                            : cdna1::kImageAtomicAddMimg,
                                           fields<cdna1::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_CDNA2:
      words = stub_image_instruction_words(cdna2::build_mimg,
                                           operation == 0   ? cdna2::kImageSampleMimg
                                           : operation == 1 ? cdna2::kImageLoadMipMimg
                                                            : cdna2::kImageAtomicAddMimg,
                                           fields<cdna2::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_RDNA1:
      words = stub_image_instruction_words(rdna1::build_mimg,
                                           operation == 0   ? rdna1::kImageSampleMimg
                                           : operation == 1 ? rdna1::kImageLoadMipMimg
                                                            : rdna1::kImageAtomicAddMimg,
                                           fields<rdna1::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_RDNA2:
      words = stub_image_instruction_words(rdna2::build_mimg,
                                           operation == 0   ? rdna2::kImageSampleMimg
                                           : operation == 1 ? rdna2::kImageLoadMipMimg
                                                            : rdna2::kImageAtomicAddMimg,
                                           fields<rdna2::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      words = stub_image_instruction_words(rdna3::build_mimg,
                                           operation == 0   ? rdna3::kImageSampleMimg
                                           : operation == 1 ? rdna3::kImageLoadMipMimg
                                                            : rdna3::kImageAtomicAddMimg,
                                           fields<rdna3::MimgBuilderFields>(2));
      break;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      words = stub_image_instruction_words(rdna3_5::build_mimg,
                                           operation == 0   ? rdna3_5::kImageSampleMimg
                                           : operation == 1 ? rdna3_5::kImageLoadMipMimg
                                                            : rdna3_5::kImageAtomicAddMimg,
                                           fields<rdna3_5::MimgBuilderFields>(2));
      break;
    default:
      FAIL();
    }
    std::unique_ptr<Decoder> decoder = Decoder::create(arch());
    DecodeResult decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    InstDefUse uses(*decoded.value(), 64);
    EXPECT_TRUE(uses.uses.contains({RegClass::SGPR, 16, 8}));
    if (operation == 0) {
      EXPECT_TRUE(uses.uses.contains({RegClass::SGPR, 12, 4}));
    }
    EXPECT_FALSE(uses.uses.contains({RegClass::SGPR, 4, 1}));
  }
}

TEST_P(AllImageAccessTest, LinearOneDimensionalArraysUseIndependentLayerStride) {
  Image image;
  image.width = 30;
  image.depth = 3;
  image.type = amdgpu::kImage1dArray;
  dimension_override_ = image.type - amdgpu::kImage1d;
  bind_image(image);
  wf_->set_exec(1);
  // Native R32 1D-array captures on gfx1100 (W7900) and gfx1201 (R9700)
  // place each layer on a 256-byte boundary.
  for (uint32_t layer : {1u, 2u}) {
    const uint32_t offset = layer * 256 + 5 * 4;
    write_coordinates(0, {5, layer, 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x12340000u + offset);
    ASSERT_TRUE(decoded_issue(false, 2));
    EXPECT_EQ(memory_.read32(kBase + offset), 0x12340000u + offset);
    memory_.write32(kBase + offset, 0xabcd0000u + offset);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(decoded_issue(true, 2));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcd0000u + offset);
  }
}

TEST_P(AllImageAccessTest, RdnaSrgbFormatsDecodeColorAndKeepLinearAlpha) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 encodes sRGB in NFMT, covered separately";
  const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
  // IEC sRGB transfer after F32 UNORM conversion. GFX11/12 point captures
  // round color channels to BF16; alpha retains its linear F32 UNORM value.
  const std::array<uint32_t, 4> expected =
      gfx10 ? std::array<uint32_t, 4>{0x3e5d0a8b, 0x3d51ffef, 0x3f800000, 0x3f008081}
            : std::array<uint32_t, 4>{0x3e5d0000, 0x3d520000, 0x3f800000, 0x3f008081};
  for (uint32_t format = 0; format < 3; ++format) {
    SCOPED_TRACE(format);
    Image image;
    auto words = rdna_descriptor(image, arch());
    util::insert_bits_into_words<uint32_t>(words.data(),
                                           arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 49 : 52,
                                           gfx10 ? 9 : 8, (gfx10 ? 128u : 64u) + format);
    util::insert_bits_into_words<uint32_t>(words.data(), 96, 12, kIdentity);
    util::insert_bits_into_words<uint32_t>(words.data(), 128, 14, (256u >> format) - 1);
    for (uint32_t i = 0; i < words.size(); ++i)
      cu_->write_sgpr(sb_ + i, words[i]);
    const uint32_t components = 1u << format;
    component_mask_ = (1u << components) - 1;
    memory_.write32(kBase, 0x80ff4080);
    wf_->set_exec(1);
    write_coordinates(0, {0, 0, 0});
    ASSERT_TRUE(decoded_issue(true, 2));
    for (uint32_t component = 0; component < components; ++component)
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + component, 0), expected[component]);
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(decoded_issue(false, 2));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(memory_.read32(kBase), 0x80ff4080u);
    EXPECT_TRUE(wf_->wait_counters().empty());
    wf_->clear_instruction_execution_error();
  }
}

TEST_P(AllImageAccessTest, ImageAtomicAddAvailabilityAndReturnValue) {
  bind_image(2);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  memory_.write32(kBase, 0x12345678);
  cu_->write_vgpr(vb_ + 8, 0, 1);
  std::array<uint32_t, 3> words{};
  switch (arch()) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    words = image_instruction_words(cdna1::build_mimg, cdna1::kImageAtomicAddMimg,
                                    fields<cdna1::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_CDNA2:
    words = image_instruction_words(cdna2::build_mimg, cdna2::kImageAtomicAddMimg,
                                    fields<cdna2::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_RDNA1: {
    auto encoding = fields<rdna1::MimgBuilderFields>(2);
    encoding.glc = 1;
    words = image_instruction_words(rdna1::build_mimg, rdna1::kImageAtomicAddMimg, encoding);
  } break;
  case ROCJITSU_CODE_ARCH_RDNA2: {
    auto encoding = fields<rdna2::MimgBuilderFields>(2);
    encoding.glc = 1;
    words = image_instruction_words(rdna2::build_mimg, rdna2::kImageAtomicAddMimg, encoding);
  } break;
  case ROCJITSU_CODE_ARCH_RDNA3:
    words = image_instruction_words(rdna3::build_mimg, rdna3::kImageAtomicAddMimg,
                                    fields<rdna3::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    words = image_instruction_words(rdna3_5::build_mimg, rdna3_5::kImageAtomicAddMimg,
                                    fields<rdna3_5::MimgBuilderFields>(2));
    break;
  case ROCJITSU_CODE_ARCH_RDNA4:
    words = rdna4::build_vimage(rdna4::kImageAtomicAddUintVimage,
                                fields<rdna4::VimageBuilderFields>(2));
    break;
  default:
    FAIL() << "Unexpected image architecture";
  }
  auto decoder = Decoder::create(arch());
  auto decoded = decoder->decode(words.data());
  ASSERT_TRUE(decoded.succeeded());
  auto instruction = std::move(decoded).value();
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4) {
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wf_).succeeded());
    const auto *info = instruction->amdgpu_memory_issue_info();
    ASSERT_NE(info, nullptr);
    ASSERT_FALSE(info->empty());
    EXPECT_EQ(info->counter_obligations()[0].wait_counter_type(),
              amdgpu::WaitCounterType::STORECNT);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(instruction.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    l2_.flush_all();
    EXPECT_EQ(memory_.read32(kBase), 0x12345679u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 1u);
    return;
  }
  instruction->execute(*instruction, wf_);
  if (arch_is_rdna(arch()) || arch() == ROCJITSU_CODE_ARCH_CDNA2) {
    ASSERT_TRUE(instruction->data());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(instruction.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    EXPECT_EQ(memory_.read32(kBase), 0x12345679u);
    if (arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2) {
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x12345678u);
    } else if (arch() == ROCJITSU_CODE_ARCH_CDNA2) {
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 1u);
    }
    return;
  }
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
  EXPECT_FALSE(instruction->data());
  EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 1u);
  EXPECT_TRUE(wf_->wait_counters().empty());
}

class Gfx11MsaaAccessTest : public ImageAccessFixture,
                            public testing::WithParamInterface<rj_code_arch_t> {
protected:
  rj_code_arch_t arch() const override { return GetParam(); }
  // Replay the physical W7900 fragment-address capture on both GFX11 targets.
  std::optional<uint32_t> image_gb_addr_config() const override { return 0x545u; }
};

INSTANTIATE_TEST_SUITE_P(Gfx11, Gfx11MsaaAccessTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5));

TEST_P(Gfx11MsaaAccessTest, LoadsFourFragmentChannelsWithPackedOrFullResults) {
  // GFX11 64 KiB R_X, 8 fragments, RGBA8_UINT. At (0,0), AddrLib
  // places the eight fragment texels consecutively in the first 32 bytes.
  Image image;
  image.width = image.height = image.pitch = 32;
  image.last_level = 3;
  image.mip_levels = 4;
  image.type = amdgpu::kImage2dMsaa;
  image.swizzle_mode = 27;
  image.rdna_format = kGfx11Rgba8Uint;
  for (uint32_t log2_samples : {1u, 2u, 3u}) {
    image.last_level = log2_samples;
    image.mip_levels = log2_samples + 1;
    auto resource = rdna_descriptor(image, arch());
    resource[3] = (resource[3] & ~0xfffu) | kIdentity;
    for (uint32_t i = 0; i < resource.size(); ++i)
      cu_->write_sgpr(sb_ + i, resource[i]);
    for (uint32_t i = 0; i < 8; ++i)
      memory_.write32(kBase + 4 * i, 0x80706050u + (i << 8));
    // W7900 returns fragments 4..7 for selectors 8 and UINT_MAX on an 8x surface.
    for (uint32_t fragment : {1u, 4u, 8u, 0xffffffffu}) {
      write_coordinates(0, {0, 0, fragment});
      write_coordinates(2, {0, 0, 0});
      for (bool packed : {false, true}) {
        fill_destination(0xdeadbeef);
        rdna3_5::MimgBuilderFields fields{};
        fields.dim = 6;
        fields.dmask = 2;
        fields.vdata = 8;
        fields.unorm = true;
        fields.d16 = packed;
        const auto words = rdna3_5::build_mimg(rdna3_5::kImageMsaaLoadMimg, fields);
        auto decoder = Decoder::create(arch());
        auto decoded = decoder->decode(words.data());
        ASSERT_TRUE(decoded.succeeded());
        auto inst = std::move(decoded).value();
        ASSERT_TRUE(inst->is_memory_op());
        inst->execute(*inst, wf_);
        ASSERT_FALSE(wf_->instruction_execution_failed());
        ASSERT_NE(inst->data(), nullptr);
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
        for (uint32_t lane : {0u, 2u}) {
          // Raw IMAGE_MSAA_LOAD on W7900 (gfx1100), RGBA8_UINT, green channel.
          constexpr std::array<uint32_t, 4> two_samples{0x60, 0x61, 0x60, 0x61};
          constexpr std::array<uint32_t, 4> low_group{0x60, 0x61, 0x62, 0x63};
          constexpr std::array<uint32_t, 4> high_group{0x64, 0x65, 0x66, 0x67};
          const auto &values = log2_samples == 1                                 ? two_samples
                               : log2_samples == 3 && lane == 0 && fragment != 1 ? high_group
                                                                                 : low_group;
          for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t expected = !packed ? values[i]
                                      : i < 2 ? values[2 * i] | (values[2 * i + 1] << 16)
                                              : 0xdeadbeef;
            EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + i, lane), expected);
            EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + i, 1), 0xdeadbeef);
          }
        }
      }
    }
  }
}

TEST_F(Rdna4ImageMemoryTest, ImplicitSampleWaitsOnContributingLanes) {
  Image image;
  image.width = image.height = 64;
  image.swizzle_mode = 1;
  image.mip_levels = 4;
  const auto bind = [&](uint32_t base, uint32_t last) {
    image.base_level = base;
    image.last_level = last;
    const auto words = rdna_descriptor(image, arch());
    for (unsigned i = 0; i < words.size(); ++i)
      cu_->write_sgpr(sb_ + i, words[i]);
  };
  auto decoder = Decoder::create(arch());
  for (unsigned level : {0u, 2u}) {
    bind(level, level);
    wf_->set_exec(1);
    write_coordinates(0, {8u >> level, 8u >> level, 0});
    cu_->write_vgpr(vb_ + 8, 0, 0x77770000u + level);
    auto store = decoder->decode_window(rdna4::build_vimage(
        rdna4::kImageStoreVimage, {.dim = 1, .dmask = 1, .vdata = 8, .vaddr0 = 0, .vaddr1 = 1}));
    ASSERT_TRUE(store.succeeded());
    ASSERT_TRUE(cu_->execute_instruction(store.value().get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(store.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
  }
  cu_->l1_vector().flush_all();
  l2_.flush_all();
  const auto bits = [](float value) { return std::bit_cast<uint32_t>(value); };
  const auto sample = rdna4::build_vsample(
      rdna4::kImageSampleVsample,
      {.dim = 1, .dmask = 1, .vdata = 8, .rsrc = 0, .samp = 8, .vaddr0 = 0, .vaddr1 = 1});
  const auto load = rdna4::build_vds(rdna4::kDsLoadB32Vds, {.addr = 20, .vdst = 0});
  wf_->set_lds_size(64);
  wf_->lds().write32(0, bits(12.5f / 64));
  uint64_t pc = 0x8000;
  for (const bool mip : {false, true}) {
    bind(0, mip ? 3 : 0);
    cu_->write_sgpr(sb_ + 8, 2 | (2 << 3) | (2 << 6));
    cu_->write_sgpr(sb_ + 9, 0x1fffu << 13);
    cu_->write_sgpr(sb_ + 10, mip ? 1u << 26 : 0);
    for (const unsigned lane : {1u, 3u}) {
      for (const bool waited : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "mip=" << mip << " lane=" << lane << " waited=" << waited);
        wf_->ensure_memory_wait_scoreboard().clear();
        wf_->clear_instruction_execution_error();
        for (unsigned i = 0; i < 4; ++i) {
          write_coordinates(i, {bits(8.5f / 64), bits(8.5f / 64), 0});
          cu_->write_vgpr(vb_ + 20, i, 0);
        }
        cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
        wf_->set_exec(1ull << lane);
        for (unsigned i = 0; i < load.size(); ++i)
          memory_.write32(pc + 4 * i, load[i]);
        uint64_t sample_pc = pc + 4 * load.size();
        if (waited) {
          memory_.write32(sample_pc, rdna4::build_sopp(rdna4::kSWaitDscntSopp)[0]);
          sample_pc += 4;
        }
        for (unsigned i = 0; i < sample.size(); ++i)
          memory_.write32(sample_pc + 4 * i, sample[i]);
        wf_->pc = pc;
        for (unsigned i = 0; i < 8 && wf_->pc != sample_pc; ++i)
          cu_->step();
        ASSERT_EQ(wf_->pc, sample_pc);
        EXPECT_EQ(cu_->read_vgpr(vb_, lane), bits(12.5f / 64));
        wf_->set_exec(1);
        const auto before = cu_->memory_wait_diagnostic_count();
        cu_->step();
        EXPECT_FALSE(wf_->instruction_execution_failed());
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x77770000u + (mip && lane == 1 ? 2 : 0));
        EXPECT_EQ(cu_->memory_wait_diagnostic_count() - before,
                  unsigned(mip && lane == 1 && !waited));
        pc += 128;
      }
    }
  }
}

TEST_F(Rdna4ImageMemoryTest, ResourceQueryTracksOnlyTheEncodedLodRegister) {
  Image image;
  image.width = 8;
  image.last_level = 1;
  image.mip_levels = 2;
  const auto decoder = Decoder::create(arch());
  auto &state = wf_->ensure_memory_wait_scoreboard();
  wf_->set_exec(1);
  for (unsigned format : {0u, kR32Uint}) {
    // RESINFO reads the descriptor dimensions and LOD even with FORMAT=INVALID.
    image.rdna_format = format;
    const auto resource = rdna_descriptor(image, arch());
    for (unsigned word = 0; word < resource.size(); ++word)
      cu_->write_sgpr(sb_ + word, resource[word]);
    for (uint8_t encoded : {5, 32}) {
      for (uint8_t mask : {0, 1}) {
        auto decoded = decoder->decode_window(rdna4::build_vimage(
            rdna4::kImageGetResinfoVimage,
            {.dim = 1, .dmask = mask, .vdata = 10, .vaddr0 = encoded, .vaddr1 = 6}));
        ASSERT_TRUE(decoded.succeeded());
        const unsigned resolved = encoded == 32 ? 0 : encoded;
        cu_->write_vgpr(vb_ + resolved, 0, 1);
        const InstDefUse uses(*decoded.value(), wf_->wf_size());
        EXPECT_EQ(uses.uses.contains({RegClass::VGPR, encoded, 1}), bool(mask));
        EXPECT_FALSE(uses.uses.contains({RegClass::VGPR, 6, 1}));
        for (uint16_t pending : {0, 5, 6, 32}) {
          state.clear();
          unsigned reports = 0;
          const auto counter = amdgpu::WaitCounterKind::Load;
          state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, pending, 1}, counter, 0xf});
          state.bind(0x200, &reports,
                     [](void *ctx, const auto &) { ++*static_cast<unsigned *>(ctx); });
          state.check_instruction(*decoded.value(), *wf_);
          EXPECT_EQ(reports, unsigned(mask && pending == resolved));
        }
        ASSERT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).succeeded());
        EXPECT_EQ(cu_->read_vgpr(vb_ + 10, 0), mask ? 4u : 0u);
      }
    }
  }
}

class Gfx11ImageFeedbackTest : public ImageAccessFixture,
                               public testing::WithParamInterface<rj_code_arch_t> {
protected:
  rj_code_arch_t arch() const override { return GetParam(); }

  void issue_image(uint16_t opcode, rdna3_5::MimgBuilderFields fields) {
    const auto words = rdna3_5::build_mimg(opcode, fields);
    memory_.write32(next_pc_, words[0]);
    memory_.write32(next_pc_ + 4, words[1]);
    wf_->pc = next_pc_;
    cu_->step();
    ASSERT_EQ(wf_->pc, next_pc_ + 8);
    ASSERT_FALSE(wf_->instruction_execution_failed());
    next_pc_ += amdgpu::InstructionCache::kLineSize;
  }

private:
  uint64_t next_pc_ = 0x4000;
};

INSTANTIATE_TEST_SUITE_P(Gfx11, Gfx11ImageFeedbackTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5));

TEST_P(Gfx11ImageFeedbackTest, PackedLoadsSignExtendOnlyTheSignedOpcode) {
  constexpr uint32_t r8_unorm = 1;
  const auto resource =
      rdna_image_descriptor(arch(), {.format = r8_unorm, .selectors = 0, .pitch = 256});
  for (uint32_t i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  wf_->set_exec(1);
  cu_->write_vgpr(vb_, 0, 0);
  cu_->write_vgpr(vb_ + 1, 0, 0);
  memory_.write32(kBase, 0x80);
  for (bool signed_load : {false, true}) {
    SCOPED_TRACE(testing::Message() << "signed_load=" << signed_load);
    cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
    ASSERT_NO_FATAL_FAILURE(
        issue_image(signed_load ? rdna3_5::kImageLoadPckSgnMimg : rdna3_5::kImageLoadPckMimg,
                    {.dim = amdgpu::kImageDim2d, .unorm = 1, .dmask = 1, .vdata = 8}));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), signed_load ? 0xffffff80u : 0x80u);
  }
}

TEST_P(Gfx11ImageFeedbackTest, OutOfBoundsAtomicSeparatesPayloadReadsFromResultOverwrites) {
  ASSERT_NO_FATAL_FAILURE(check_out_of_bounds_atomic_payload());
}

TEST_P(Gfx11ImageFeedbackTest, AtomicTfeReportsOnlyTheNonresidentReturningLane) {
  for (uint32_t level : {0u, 1u})
    ASSERT_NO_FATAL_FAILURE(check_atomic_prt_access(/*compare_swap=*/false, level));
}

TEST_P(Gfx11ImageFeedbackTest, CompareSwapTfeWritesStatusOverFirstComparisonWordOnANack) {
  for (uint32_t level : {0u, 1u})
    ASSERT_NO_FATAL_FAILURE(check_atomic_prt_access(/*compare_swap=*/true, level));
}

TEST_P(Gfx11ImageFeedbackTest, ImplicitDerivativesCheckOnlyContributingHelperCoordinates) {
  Image image;
  image.width = image.height = 4;
  image.pitch = 1;
  image.mip_levels = 3;
  image.last_level = 2;
  const auto resource = rdna_descriptor(image, arch());
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  cu_->write_sgpr(sb_ + 9, 0xfffu << 12); // maximum sample LOD
  for (unsigned mip = 0; mip <= 2; ++mip) {
    const auto layout = amdgpu::ImageAddressing::prepare(
        cu_->image_address_config(), {.width = 4, .height = 4, .element_bytes = 4, .mip_levels = 3},
        mip);
    ASSERT_TRUE(layout);
    memory_.write32(kBase + layout->byte_offset(0, 0, 0), 100 + mip);
  }
  const auto query = rdna3_5::kImageGetLodMimg;
  const auto gradients = rdna3_5::kImageSampleDMimg;
  const auto sample = rdna3_5::kImageSampleMimg;
  struct Case {
    uint16_t opcode;
    uint8_t reg, producer_lane, consumer_lane, lod;
    bool hazard, a16 = false, nsa = false, mip_filter = true;
  };
  const Case cases[]{
      {gradients, 2, 1, 0, 0, false}, // Explicit dX/dx at helper lane.
      {gradients, 6, 1, 0, 0, false}, // Explicit-gradient sample's X.
      {gradients, 2, 0, 0, 2, true},
      {sample, 2, 1, 0, 2, true},                       // Ordinary descriptor, implicit mip.
      {sample, 2, 1, 0, 0, false, false, false, false}, // Mip filtering disabled.
  };
  wf_->set_lds_size(64);
  uint64_t pc = 0x8000;
  constexpr uint32_t sentinel = 0xdeadbeef;
  for (const auto &test : cases) {
    cu_->write_sgpr(sb_ + 10, test.mip_filter ? 1u << 26 : 0);
    for (bool waited : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "opcode=" << test.opcode << " v" << +test.reg << " producer lane="
                   << +test.producer_lane << " consumer lane=" << +test.consumer_lane
                   << " A16=" << test.a16 << " NSA=" << test.nsa << " waited=" << waited);
      auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
      scoreboard.clear();
      for (unsigned lane = 0; lane < 8; ++lane) {
        for (unsigned reg : {2u, 3u, 4u, 5u, 6u, 7u, 12u})
          cu_->write_vgpr(vb_ + reg, lane, 0);
        cu_->write_vgpr(vb_ + 8, lane, sentinel);
        cu_->write_vgpr(vb_ + 9, lane, sentinel);
      }
      wf_->lds().write32(0, test.a16 ? 0x3c000000u : std::bit_cast<uint32_t>(1.0f));
      const auto load = rdna3_5::build_ds(rdna3_5::kDsLoadB32Ds, {.addr = 20, .vdst = test.reg});
      memory_.write32(pc, load[0]);
      memory_.write32(pc + 4, load[1]);
      uint64_t next = pc + 8;
      if (waited) {
        memory_.write32(next, rdna3_5::build_sopp(rdna3_5::kSWaitcntSopp)[0]);
        next += 4;
      }
      memory_.write32(next, rdna3_5::build_sop1(
                                rdna3_5::kSMovB32Sop1,
                                {.ssrc0 = static_cast<uint8_t>(128 + (1u << test.consumer_lane)),
                                 .sdst = 126})[0]);
      next += 4;
      const bool is_query = test.opcode == query;
      const auto words =
          rdna3_5::build_mimg(test.opcode, {.nsa = test.nsa,
                                            .dim = 1,
                                            .dmask = static_cast<uint8_t>(is_query ? 3 : 1),
                                            .a16 = test.a16,
                                            .vaddr = 2,
                                            .vdata = 8,
                                            .ssamp = 2});
      memory_.write32(next, words[0]);
      memory_.write32(next + 4, words[1]);
      if (test.nsa)
        memory_.write32(next + 8, 12);
      wf_->pc = pc;
      wf_->set_exec(uint64_t{1} << test.producer_lane);
      const auto before = cu_->memory_wait_diagnostic_count();
      for (unsigned step = 0; step < 8 && wf_->pc != next; ++step)
        cu_->step();
      ASSERT_EQ(wf_->pc, next);
      ASSERT_EQ(wf_->exec(), uint64_t{1} << test.consumer_lane);
      ASSERT_EQ(cu_->memory_wait_diagnostic_count(), before);
      cu_->step();
      ASSERT_EQ(wf_->pc, next + (test.nsa ? 12 : 8));
      ASSERT_FALSE(wf_->instruction_execution_failed());
      EXPECT_EQ(cu_->memory_wait_diagnostic_count() - before, test.hazard && !waited ? 1u : 0u);
      const unsigned result_reg = is_query ? 9 : 8;
      const auto result =
          is_query ? std::bit_cast<uint32_t>(static_cast<float>(test.lod)) : 100u + test.lod;
      for (unsigned lane = 0; lane < 8; ++lane)
        EXPECT_EQ(cu_->read_vgpr(vb_ + result_reg, lane),
                  lane == test.consumer_lane ? result : sentinel)
            << lane;
      pc += amdgpu::InstructionCache::kLineSize;
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, PackedArrayCoordinateWaitsSeparateLayerAndGradientLanes) {
  auto decoder = Decoder::create(arch());
  auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
  wf_->set_exec(1);
  cu_->write_sgpr(sb_ + 10, 1u << 26); // nearest mip
  for (bool mip_chain : {false, true}) {
    Image image;
    image.type = amdgpu::kImage1dArray;
    image.width = 4;
    image.depth = 2;
    image.last_level = mip_chain ? 2 : 0;
    image.mip_levels = image.last_level + 1;
    const auto resource = rdna_descriptor(image, arch());
    for (unsigned i = 0; i < resource.size(); ++i)
      cu_->write_sgpr(sb_ + i, resource[i]);
    for (bool query : {false}) {
      const auto words =
          rdna3_5::build_mimg(query ? rdna3_5::kImageGetLodMimg : rdna3_5::kImageSampleMimg,
                              {.dim = 4, .dmask = 1, .a16 = 1, .vaddr = 2, .vdata = 8, .ssamp = 2});
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      for (unsigned lane = 0; lane < 4; ++lane) {
        for (uint8_t bytes : {0x3, 0xc}) {
          SCOPED_TRACE(testing::Message() << "query=" << query << " mip chain=" << mip_chain
                                          << " lane=" << lane << " bytes=" << +bytes);
          unsigned reports = 0;
          scoreboard.clear();
          const auto counter = amdgpu::WaitCounterKind::Ds;
          scoreboard.add({scoreboard.issue(counter),
                          0x100,
                          uint64_t{1} << lane,
                          {RegClass::VGPR, 2, 1},
                          counter,
                          bytes});
          scoreboard.bind(0x200, &reports,
                          [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          scoreboard.check_instruction(*decoded.value(), *wf_);
          // U is low, the array layer is high. Only U supplies derivatives;
          // GET_LOD does not select a layer even for an active lane.
          const bool read = bytes == 0xc ? (!query && lane == 0)
                                         : lane == 0 || ((query || mip_chain) && lane < 3);
          EXPECT_EQ(reports, unsigned(read));
        }
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, WaitsCheckOnlyTheImageStorePayloadWordsFetched) {
  const auto resource =
      rdna_image_descriptor(arch(), {.format = kRgba32Uint, .selectors = kIdentity});
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  wf_->set_lds_size(64);
  wf_->lds().write32(0, 0x5a7e1234);
  cu_->write_vgpr(vb_ + 20, 0, 0);
  uint64_t pc = 0x8000;
  struct Case {
    uint8_t vdata, pending;
    bool waited, out_of_bounds, hazard;
  };
  const Case cases[] = {
      {31, 0, false, false, true},  {31, 0, true, false, false},  {30, 0, false, false, false},
      {30, 30, false, false, true}, {31, 31, false, true, false},
  };
  for (bool r128 : {false, true}) {
    for (const auto &test : cases) {
      SCOPED_TRACE(testing::Message()
                   << "r128=" << r128 << " vdata=" << +test.vdata << " pending=" << +test.pending
                   << " waited=" << test.waited << " out_of_bounds=" << test.out_of_bounds);
      wf_->ensure_memory_wait_scoreboard().clear();
      wf_->set_exec(1);
      cu_->write_vgpr(vb_ + 2, 0, test.out_of_bounds ? 3 : 0);
      cu_->write_vgpr(vb_ + 3, 0, 0);
      cu_->write_vgpr(vb_ + 0, 0, 0x33445566);
      cu_->write_vgpr(vb_ + 30, 0, 0x11112222);
      cu_->write_vgpr(vb_ + 31, 0, 0x77778888);
      const auto load =
          rdna3_5::build_ds(rdna3_5::kDsLoadB32Ds, {.addr = 20, .vdst = test.pending});
      memory_.write32(pc, load[0]);
      memory_.write32(pc + 4, load[1]);
      uint64_t store_pc = pc + 8;
      if (test.waited) {
        memory_.write32(store_pc, rdna3_5::build_sopp(rdna3_5::kSWaitcntSopp)[0]);
        store_pc += 4;
      }
      const auto store = rdna3_5::build_mimg(
          rdna3_5::kImageStoreMimg,
          {.dim = 1, .unorm = true, .dmask = 3, .r128 = r128, .vaddr = 2, .vdata = test.vdata});
      memory_.write32(store_pc, store[0]);
      memory_.write32(store_pc + 4, store[1]);
      wf_->pc = pc;
      const auto before = cu_->memory_wait_diagnostic_count();
      for (unsigned step = 0; step < 8 && wf_->pc != store_pc; ++step)
        cu_->step();
      ASSERT_EQ(wf_->pc, store_pc);
      ASSERT_EQ(cu_->memory_wait_diagnostic_count(), before);
      cu_->step();
      EXPECT_EQ(wf_->pc, store_pc + 8);
      EXPECT_FALSE(wf_->instruction_execution_failed());
      EXPECT_EQ(cu_->memory_wait_diagnostic_count() - before, unsigned(test.hazard));
      if (wf_->instruction_execution_failed())
        return;
      if (!test.out_of_bounds) {
        cu_->l1_vector().flush_all();
        l2_.flush_all();
        const uint32_t source = test.vdata == 31 ? 0 : 31;
        EXPECT_EQ(memory_.read32(kBase + 4), cu_->read_vgpr(vb_ + source, 0));
      }
      pc += amdgpu::InstructionCache::kLineSize;
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, ImageStoresReadIndividualAllocatedAndFallbackDataWords) {
  wf_->set_exec(1);
  const auto resource =
      rdna_image_descriptor(arch(), {.format = kRgba32Uint, .selectors = kIdentity});
  for (uint32_t i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  constexpr uint32_t fallback = 0x00440033;
  constexpr uint32_t first = 0x00220011;
  constexpr uint32_t second = 0x00660055;
  cu_->write_vgpr(vb_, 0, fallback);
  cu_->write_vgpr(vb_ + 2, 0, 0);
  cu_->write_vgpr(vb_ + 3, 0, 0);
  // The ordinary descriptor and its R128 form use separate image preparation paths.
  for (const bool r128 : {false, true}) {
    for (const bool d16 : {false, true}) {
      for (const uint8_t data : {30, 31, 32}) {
        SCOPED_TRACE(testing::Message()
                     << "r128=" << r128 << " d16=" << d16 << " vdata=" << unsigned(data));
        if (data < 32)
          cu_->write_vgpr(vb_ + data, 0, first);
        if (data == 30)
          cu_->write_vgpr(vb_ + 31, 0, second);
        ASSERT_NO_FATAL_FAILURE(
            issue_image(rdna3_5::kImageStoreMimg, {.dim = 1,
                                                   .unorm = true,
                                                   .dmask = uint8_t(d16 ? 15 : 3),
                                                   .r128 = r128,
                                                   .d16 = d16,
                                                   .vaddr = 2,
                                                   .vdata = data}));
        cu_->l1_vector().flush_all();
        l2_.flush_all();
        const uint32_t x = data < 32 ? first : fallback;
        const uint32_t y = data == 30 ? second : fallback;
        const std::array<uint32_t, 4> expected =
            d16 ? std::array<uint32_t, 4>{x & 0xffffu, x >> 16, y & 0xffffu, y >> 16}
                : std::array<uint32_t, 4>{x, y, 0, 0};
        for (unsigned channel = 0; channel < expected.size(); ++channel)
          EXPECT_EQ(memory_.read32(kBase + 4 * channel), expected[channel]) << channel;
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, ImageTransfersAndSamplesResolveEachCoordinateWord) {
  wf_->set_exec(1);
  const auto resource = rdna_image_descriptor(arch(), {.width = 4, .height = 2});
  for (uint32_t i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  for (uint32_t i = 0; i < 4; ++i)
    cu_->write_sgpr(sb_ + 8 + i, 0);                             // Point sampling at level zero.
  constexpr uint64_t texel = kBase + 256 + 2 * sizeof(uint32_t); // (2, 1)
  memory_.write32(kBase + 2 * sizeof(uint32_t), 19);
  for (const uint8_t address : {30, 31}) {
    for (const auto opcode :
         {rdna3_5::kImageLoadMimg, rdna3_5::kImageSampleLzMimg, rdna3_5::kImageAtomicCmpswapMimg}) {
      SCOPED_TRACE(testing::Message() << "vaddr=" << unsigned(address) << " opcode=" << opcode);
      const bool sample = opcode == rdna3_5::kImageSampleLzMimg;
      const bool atomic = opcode == rdna3_5::kImageAtomicCmpswapMimg;
      const uint32_t x = sample ? std::bit_cast<uint32_t>(2.5f) : 2u;
      const uint32_t y = sample ? std::bit_cast<uint32_t>(1.5f) : 1u;
      cu_->write_vgpr(vb_, 0, address == 31 ? y : 0);
      cu_->write_vgpr(vb_ + address, 0, x);
      if (address == 30)
        cu_->write_vgpr(vb_ + 31, 0, y);
      cu_->write_vgpr(vb_ + 8, 0, 7);
      cu_->write_vgpr(vb_ + 9, 0, 107);
      memory_.write32(texel, 107);
      ASSERT_NO_FATAL_FAILURE(issue_image(opcode, {.dim = 1,
                                                   .unorm = true,
                                                   .dmask = uint8_t(atomic ? 3 : 1),
                                                   .glc = atomic,
                                                   .vaddr = address,
                                                   .vdata = 8,
                                                   .ssamp = uint8_t(sample ? 2 : 0)}));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 107u);
      cu_->l1_vector().flush_all();
      l2_.flush_all();
      if (atomic) {
        EXPECT_EQ(memory_.read32(texel), 7u);
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, ResourceInfoReadsLodFromVgprZeroForAnUnallocatedSelector) {
  wf_->set_exec(1);
  Image image;
  image.width = 8;
  image.mip_levels = 2;
  image.last_level = 1;
  const auto resource = rdna_descriptor(image, arch());
  for (uint32_t i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  cu_->write_vgpr(vb_, 0, 1);
  for (const uint8_t address : {0, 32}) {
    cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
    ASSERT_NO_FATAL_FAILURE(issue_image(rdna3_5::kImageGetResinfoMimg,
                                        {.dim = 1, .dmask = 1, .vaddr = address, .vdata = 8}));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 4u);
  }
}

TEST_P(Gfx11ImageFeedbackTest, ReturningAtomicsValidateTheWholeResultIndependentlyOfSources) {
  enum class Boundary { StatusFits, StatusOutside, NoTfe, NoReturn };
  uint64_t pc = 0x4000;
  wf_->set_exec(1);
  for (const bool wide : {false, true}) {
    Image image;
    image.rdna_format = wide ? 48 : kR32Uint; // 32_32_UINT and 32_UINT
    const auto resource = rdna_descriptor(image, arch());
    for (uint32_t i = 0; i < resource.size(); ++i)
      cu_->write_sgpr(sb_ + i, resource[i]);
    const unsigned texel_words = wide ? 2 : 1;
    const uint64_t old = wide ? (uint64_t{107} << 32) | 107 : 107;
    for (const bool compare_swap : {false, true}) {
      for (const auto boundary :
           {Boundary::StatusFits, Boundary::StatusOutside, Boundary::NoTfe, Boundary::NoReturn}) {
        SCOPED_TRACE(testing::Message() << "wide=" << wide << " cmpswap=" << compare_swap
                                        << " boundary=" << static_cast<int>(boundary));
        const bool suppressed = boundary == Boundary::StatusOutside;
        const bool returns = boundary != Boundary::NoReturn;
        const uint8_t data = 32 - texel_words - (boundary == Boundary::StatusFits);
        const unsigned input_words = texel_words * (compare_swap ? 2 : 1);
        const auto words = rdna3_5::build_mimg(
            compare_swap ? rdna3_5::kImageAtomicCmpswapMimg : rdna3_5::kImageAtomicAddMimg,
            {.dim = 1,
             .unorm = 1,
             .dmask = static_cast<uint8_t>((1u << input_words) - 1),
             .glc = returns,
             .vaddr = 2,
             .vdata = data,
             .tfe = boundary != Boundary::NoTfe});
        memory_.write64(kBase, old);
        cu_->write_vgpr(vb_, 0, 107); // Any missing comparison word selects v0.
        for (unsigned reg = data; reg < 32; ++reg)
          cu_->write_vgpr(vb_ + reg, 0, 0xdeadbeef);
        for (unsigned word = 0; word < input_words && data + word < 32; ++word)
          cu_->write_vgpr(vb_ + data + word, 0, word >= texel_words ? 107 : word == 0 ? 7 : 0);
        const uint32_t status = data + texel_words;
        const uint32_t before_status = status < 32 ? cu_->read_vgpr(vb_ + status, 0) : 0;
        memory_.write32(pc, words[0]);
        memory_.write32(pc + 4, words[1]);
        wf_->pc = pc;
        cu_->step();
        EXPECT_EQ(wf_->pc, pc + 8);
        EXPECT_EQ(memory_.read64(kBase), suppressed ? old : compare_swap ? 7u : old + 7);
        EXPECT_EQ(cu_->read_vgpr(vb_ + data, 0), returns && !suppressed ? 107u : 7u);
        if (wide) {
          EXPECT_EQ(cu_->read_vgpr(vb_ + data + 1, 0), returns && !suppressed ? 107u : 0u);
        }
        if (status < 32) {
          EXPECT_EQ(cu_->read_vgpr(vb_ + status, 0), before_status);
        }
        // Each instruction is written after the previous one has issued.
        // Start on a fresh I-cache line so the earlier fetch cannot cache zeros.
        pc += amdgpu::InstructionCache::kLineSize;
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, R128ControlsScalarResourceReadsAcrossMimgForms) {
  auto decoder = Decoder::create(arch());
  auto &state = wf_->ensure_memory_wait_scoreboard();
  for (uint16_t opcode :
       {rdna3_5::kImageLoadMimg, rdna3_5::kImageStoreMimg, rdna3_5::kImageAtomicAddMimg,
        rdna3_5::kImageGetResinfoMimg, rdna3_5::kImageBvhIntersectRayMimg,
        rdna3_5::kImageBvh64IntersectRayMimg}) {
    const bool bvh = opcode == rdna3_5::kImageBvhIntersectRayMimg ||
                     opcode == rdna3_5::kImageBvh64IntersectRayMimg;
    const bool sampler =
        opcode == rdna3_5::kImageSampleLzOMimg || opcode == rdna3_5::kImageGetLodMimg;
    for (bool r128 : {false, true}) {
      for (bool nsa : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "opcode=" << opcode << " r128=" << r128 << " nsa=" << nsa);
        rdna3_5::MimgBuilderFields fields{};
        fields.dim = bvh ? 0 : 1;
        fields.unorm = true;
        fields.nsa = nsa;
        fields.dmask = bvh ? 15 : 1;
        fields.r128 = r128;
        fields.vaddr = 8;
        fields.ssamp = sampler ? 3 : 0;
        const auto base = rdna3_5::build_mimg(opcode, fields);
        const uint32_t words[] = {base[0], base[1], 0x12100e0c};
        auto decoded = decoder->decode_window(words);
        ASSERT_TRUE(decoded.succeeded());
        const auto &inst = *decoded.value();
        const InstDefUse du(inst, wf_->wf_size());
        // Ordinary LLVM syntax spells eight SGPRs even with the narrower read.
        EXPECT_NE(inst.disassemble().find(bvh ? "s[0:3]" : "s[0:7]"), std::string::npos);
        for (uint16_t reg : {0, 3, 4, 7, 8, 12, 15, 16}) {
          const RegisterRef ref{RegClass::SGPR, reg, 1};
          const bool used = reg < (bvh || r128 ? 4u : 8u) || (sampler && reg >= 12 && reg <= 15);
          EXPECT_EQ(du.uses.contains(ref), used) << reg;
          unsigned reports = 0;
          state.clear();
          const auto counter = amdgpu::WaitCounterKind::Ds;
          state.add({state.issue(counter), 0x100, 1, ref, counter, 0xf});
          state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          state.check_instruction(inst, *wf_);
          EXPECT_EQ(reports, unsigned(used)) << reg;
        }
        if (bvh && !r128) {
          wf_->clear_instruction_execution_error();
          EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
          EXPECT_EQ(wf_->instruction_execution_error(),
                    amdgpu::InstructionExecutionError::UnsupportedOperandValue);
        }
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, QueriesSuppressTheEntireOutOfRangeRegisterTuple) {
  // A null resource returns zero for every enabled query component.
  for (uint32_t i = 0; i < 8; ++i)
    cu_->write_sgpr(sb_ + i, 0);
  auto decoder = Decoder::create(arch());
  constexpr uint32_t sentinel = 0xdeadbeef;
  for (auto opcode : {rdna3_5::kImageGetResinfoMimg}) {
    for (uint8_t dst : {30, 31}) {
      SCOPED_TRACE(testing::Message() << opcode << " v" << unsigned(dst));
      rdna3_5::MimgBuilderFields fields{};
      fields.dim = 1;
      fields.dmask = 3;
      fields.vdata = dst;
      for (unsigned lane = 0; lane < 3; ++lane)
        for (unsigned reg : {30u, 31u})
          cu_->write_vgpr(vb_ + reg, lane, sentinel);
      auto words = rdna3_5::build_mimg(opcode, fields);
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      ASSERT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).succeeded());
      for (unsigned lane = 0; lane < 3; ++lane)
        for (unsigned reg : {30u, 31u})
          EXPECT_EQ(cu_->read_vgpr(vb_ + reg, lane), dst == 30 && lane != 1 ? 0u : sentinel);
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, DroppedSamplesWaitOnlyForTheResultsTheyWrite) {
  class ScalarReads final : public ExecutionPlugin {
  public:
    ScalarReads() : ExecutionPlugin("sample_scalar_reads") {}
    void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *, RegisterRef reg) override {
      if (reg.cls == RegClass::SGPR && (reg.index == 0 || reg.index == 8))
        mask |= 1u << reg.index;
    }
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t) override {
      if (reg == wave->vgpr_alloc().base)
        v0 |= lanes;
    }
    uint64_t v0 = 0;
    unsigned mask = 0;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<ScalarReads>();
  auto *reads = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  const auto resource = rdna_image_descriptor(arch(), {.format = 22});
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  wf_->set_exec(1);
  auto decoder = Decoder::create(arch());
  auto &score = wf_->ensure_memory_wait_scoreboard();
  for (auto [mask, dst] :
       std::array<std::pair<uint8_t, uint8_t>, 4>{{{0, 8}, {1, 8}, {3, 30}, {3, 31}}})
    for (bool tfe : {false, true})
      for (bool lwe : {false, true}) {
        SCOPED_TRACE(testing::Message() << unsigned(mask) << " dst=" << unsigned(dst)
                                        << " TFE=" << tfe << " LWE=" << lwe);
        auto decoded =
            decoder->decode_window(rdna3_5::build_mimg(rdna3_5::kImageSampleLzMimg, {.dim = 1,
                                                                                     .unorm = 1,
                                                                                     .dmask = mask,
                                                                                     .vdata = dst,
                                                                                     .tfe = tfe,
                                                                                     .lwe = lwe,
                                                                                     .ssamp = 2}));
        ASSERT_TRUE(decoded.succeeded());
        score.clear();
        constexpr auto counter = amdgpu::WaitCounterKind::Ds;
        for (const auto reg : {RegisterRef{RegClass::SGPR, 0, 1}, RegisterRef{RegClass::SGPR, 8, 1},
                               RegisterRef{RegClass::VGPR, dst, 1}})
          score.add({score.issue(counter), 0x100, 1, reg, counter, 0xf});
        struct Waits {
          unsigned scalars = 0;
          bool destination = false;
        } waits;
        score.bind(0x200, &waits, [](void *data, const auto &hazard) {
          auto &waits = *static_cast<Waits *>(data);
          if (hazard.write)
            waits.destination = true;
          else if (hazard.reg.cls == RegClass::SGPR)
            waits.scalars |= 1u << hazard.reg.index;
        });
        score.check_instruction(*decoded.value(), *wf_);
        score.clear();
        reads->mask = 0;
        reads->v0 = 0;
        ASSERT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).succeeded());
        const bool fits = dst + (mask == 3 ? 2u : 1u) + unsigned(tfe || lwe) <= 32;
        const bool acquires_sources = fits && (mask || lwe);
        EXPECT_EQ(reads->mask, acquires_sources ? 0x101u : 0u);
        EXPECT_EQ(bool(reads->v0), acquires_sources);
        EXPECT_EQ(waits.scalars, reads->mask);
        EXPECT_EQ(waits.destination, fits);
      }
}

TEST_P(Gfx11ImageFeedbackTest, DroppedFetchReservesTfeAndResourceQueryReturnsZero) {
  auto decoder = Decoder::create(arch());
  constexpr uint32_t sentinel = 0xdeadbeef;
  for (const auto opcode : {rdna3_5::kImageLoadMimg, rdna3_5::kImageGetResinfoMimg}) {
    for (bool tfe : {false, true}) {
      const auto words =
          rdna3_5::build_mimg(opcode, {.dim = 1, .dmask = 0, .vdata = 31, .tfe = tfe});
      auto decoded = decoder->decode_window(words);
      ASSERT_TRUE(decoded.succeeded());
      auto inst = std::move(decoded).value();
      EXPECT_NE(inst->disassemble().find(tfe ? " v[31:32]," : " v31,"), std::string::npos);
      for (unsigned lane : {0u, 1u, 2u})
        cu_->write_vgpr(vb_ + 31, lane, sentinel);
      ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
      if (opcode == rdna3_5::kImageLoadMimg) {
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      }
      for (unsigned lane : {0u, 1u, 2u}) {
        const bool suppressed = opcode == rdna3_5::kImageLoadMimg && tfe;
        EXPECT_EQ(cu_->read_vgpr(vb_ + 31, lane), suppressed || lane == 1 ? sentinel : 0u);
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, GatherAndMsaaRequireOneSelectedComponent) {
  auto decoder = Decoder::create(arch());
  for (const auto opcode : {rdna3_5::kImageGather4LzMimg, rdna3_5::kImageMsaaLoadMimg}) {
    for (uint8_t mask : {0u, 3u}) {
      for (const bool tfe : {false, true}) {
        const auto words = rdna3_5::build_mimg(
            opcode, {.dim = static_cast<uint8_t>(opcode == rdna3_5::kImageMsaaLoadMimg ? 6 : 1),
                     .dmask = mask,
                     .vdata = 8,
                     .tfe = tfe});
        auto decoded = decoder->decode_window(words);
        ASSERT_TRUE(decoded.succeeded());
        wf_->clear_instruction_execution_error();
        EXPECT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).failed());
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnsupportedOperandValue);
      }
    }
  }
}

class Gfx11MsaaFeedbackTest : public Gfx11ImageFeedbackTest {
protected:
  std::optional<uint32_t> image_gb_addr_config() const override { return 0x545u; }
};

INSTANTIATE_TEST_SUITE_P(Gfx11, Gfx11MsaaFeedbackTest,
                         testing::Values(ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5));

TEST_P(Gfx11MsaaFeedbackTest, MsaaFragmentsDriveRoutedFootprintsAndWatchpoints) {
  class Footprints final : public ExecutionPlugin {
  public:
    Footprints() : ExecutionPlugin("image_fragments") {}
    bool observes_memory_routing() const override { return true; }
    void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access) override {
      EXPECT_TRUE(access.addresses.empty());
      for (const auto &tap : access.image_sample_footprints) {
        EXPECT_EQ(tap.lane_mask, 1u);
        fragments.push_back(tap.addresses[0]);
      }
    }
    std::vector<uint64_t> fragments;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Footprints>();
  auto *footprints = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  Image image;
  image.width = image.height = image.pitch = 32;
  image.last_level = 3;
  image.mip_levels = 4;
  image.type = amdgpu::kImage2dMsaa;
  image.swizzle_mode = 27;
  image.rdna_format = kGfx11Rgba8Uint;
  auto resource = rdna_descriptor(image, arch());
  resource[3] = (resource[3] & ~0xfffu) | kIdentity;
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  for (unsigned i = 0; i < 8; ++i)
    memory_.write32(kBase + 4 * i, 0x80706050u + (i << 8));
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 4});
  std::vector<uint64_t> watched;
  cu_->set_debug_active(true);
  cu_->set_watchpoint_handler([&](amdgpu::Wavefront &, uint64_t addr, uint32_t bytes, bool, bool) {
    EXPECT_EQ(bytes, 4u);
    watched.push_back(addr);
    return false;
  });
  ASSERT_NO_FATAL_FAILURE(
      issue_image(rdna3_5::kImageMsaaLoadMimg, {.dim = 6, .unorm = 1, .dmask = 2, .vdata = 8}));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x64u);
  const std::vector<uint64_t> expected{kBase + 16, kBase + 20, kBase + 24, kBase + 28};
  EXPECT_EQ(watched, expected);
  EXPECT_EQ(footprints->fragments, expected);
}

TEST_P(Gfx11ImageFeedbackTest, DebuggerDistinguishesSparseTextureMissesFromVmFaults) {
  class MissingPage final : public amdgpu::AddressSpaceTranslator {
  public:
    bool published = true;
    std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
      static const auto holes = std::make_shared<const amdgpu::PrtRanges>(
          amdgpu::PrtRanges{{kBase + 4096, kBase + 8192}});
      return published ? holes : nullptr;
    }

  private:
    amdgpu::IdentityAddressSpaceTranslator identity_;
    amdgpu::VmTranslationResult translate(uint64_t addr, size_t size,
                                          amdgpu::VmAccessKind access) const override {
      if ((addr >> 12) == (kBase + 4096) >> 12)
        return {};
      return identity_.translate(addr, size, access);
    }
  };
  amdgpu::GpuVm vm;
  auto pages = std::make_shared<MissingPage>();
  auto handle = vm.register_address_space(
      1, pages, std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_), {});
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_exec(1);
  memory_.write32(kBase, 0x12345678);
  cu_->set_debug_active(true);
  std::vector<uint64_t> faults, watched;
  cu_->set_memory_violation_handler([&](amdgpu::Wavefront &, uint64_t addr, bool, uint64_t) {
    faults.push_back(addr);
    return true;
  });
  cu_->set_watchpoint_handler([&](amdgpu::Wavefront &, uint64_t addr, uint32_t, bool, bool) {
    watched.push_back(addr);
    return false;
  });
  for (bool sparse : {false, true}) {
    auto resource = rdna_descriptor(Image{.width = 2048, .pitch = 2048}, arch());
    pages->published = sparse;
    for (unsigned i = 0; i < resource.size(); ++i)
      cu_->write_sgpr(sb_ + i, resource[i]);
    for (bool resident : {true, false}) {
      SCOPED_TRACE(testing::Message() << "sparse=" << sparse << " resident=" << resident);
      faults.clear();
      watched.clear();
      fill_destination(0xdeadbeef);
      write_coordinates(0, {resident ? 0u : 1024u, 0, 0});
      ASSERT_NO_FATAL_FAILURE(issue_image(
          rdna3_5::kImageLoadMimg, {.dim = 1, .unorm = 1, .dmask = 1, .vdata = 8, .tfe = sparse}));
      const bool fault = !sparse && !resident;
      EXPECT_EQ(faults, fault ? std::vector<uint64_t>{kBase + 4096} : std::vector<uint64_t>{});
      EXPECT_EQ(watched, resident ? std::vector<uint64_t>{kBase} : std::vector<uint64_t>{});
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), fault ? 0xdeadbeefu : resident ? 0x12345678u : 0u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), sparse && !resident ? 1u : 0xdeadbeefu);
    }
  }
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(Gfx11ImageFeedbackTest, GatherFloatPayloadAcrossLayoutsAndFeedback) {
  wf_->set_exec(1);
  for (unsigned swizzle : {0u, 27u}) {
    Image image{.width = 2, .height = 2, .pitch = 64, .swizzle_mode = swizzle, .rdna_format = 22};
    for (unsigned threshold : {0u, 256u}) {
      auto resource = rdna_descriptor(image, arch());
      resource[5] |= threshold << 8;
      for (unsigned i = 0; i < resource.size(); ++i)
        cu_->write_sgpr(sb_ + i, resource[i]);
      cu_->write_sgpr(sb_ + 8, 2 | (2 << 3) | (1 << 15));
      for (uint32_t payload : {0x7fc12345u, 0x80000001u, 0x3f800000u}) {
        for (unsigned offset = 0; offset < 65536; offset += 4)
          memory_.write32(kBase + offset, payload);
        cu_->l1_vector().invalidate_all();
        l2_.invalidate_all();
        write_coordinates(0, {0x3f800000u, 0x3f800000u, 0});
        for (bool lwe : {false, true}) {
          for (uint16_t opcode : {rdna3_5::kImageGather4LzMimg}) {
            SCOPED_TRACE(testing::Message()
                         << "swizzle=" << swizzle << " threshold=" << threshold << " lwe=" << lwe
                         << " opcode=" << opcode << " payload=" << std::hex << payload);
            fill_destination(0xdeadbeef);
            cu_->write_vgpr(vb_ + 12, 0, 0xdeadbeefu);
            ASSERT_NO_FATAL_FAILURE(issue_image(
                opcode, {.dim = 1, .unorm = 1, .dmask = 1, .vdata = 8, .lwe = lwe, .ssamp = 2}));
            for (unsigned reg = 8; reg < 12; ++reg)
              EXPECT_EQ(cu_->read_vgpr(vb_ + reg, 0), payload);
            // Without mip filtering the LWE bit does not report a warning.
            EXPECT_EQ(cu_->read_vgpr(vb_ + 12, 0), 0xdeadbeefu);
          }
        }
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, LodWarningsMatchTheSampleAndSetStickyMode) {
  Image image;
  image.width = image.height = 64;
  image.pitch = 1; // The pitch field is zero for a mip chain.
  image.last_level = 3;
  image.mip_levels = 4;
  image.rdna_format = 22; // R32_FLOAT, valid for linear mip filtering.
  auto resource = rdna_descriptor(image, arch());
  resource[5] |= 384u << 8; // Warn below absolute LOD 1.5.
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  cu_->write_sgpr(sb_ + 8, 2 | (2 << 3));
  auto decoder = Decoder::create(arch());
  const auto read_mode = decoder->decode_window(
      rdna3_5::build_sopk(rdna3_5::kSGetregB32Sopk, {.simm16 = 1 | (10 << 6), .sdst = 16}));
  ASSERT_TRUE(read_mode.succeeded());
  constexpr uint32_t initial = 0xdeadbeefu;
  struct Case {
    uint16_t opcode;
    uint8_t mip_filter;
    float lod;
    unsigned minimum;
    bool lwe, active;
    uint32_t status;
    bool clear_mode, sticky;
  };
  for (const auto &test : {
           Case{rdna3_5::kImageSampleLMimg, 0, 0.5f, 0, true, true, initial, true, false},
           Case{rdna3_5::kImageSampleLMimg, 1, 0.5f, 0, false, true, initial, true, false},
           Case{rdna3_5::kImageSampleLMimg, 1, 0.5f, 0, true, false, initial, true, false},
           Case{rdna3_5::kImageSampleLMimg, 2, 0.5f, 384, true, true, initial, true, false},
           Case{rdna3_5::kImageSampleLMimg, 1, 0.5f, 0, true, true, 0x10002, true, true},
           Case{rdna3_5::kImageSampleLMimg, 2, 0.5f, 0, true, true, 2, true, true},
           Case{rdna3_5::kImageSampleLMimg, 1, 2.0f, 0, true, true, initial, false, true},
           Case{rdna3_5::kImageGather4LzMimg, 1, 0.0f, 320, true, true, 2, true, true},
       }) {
    SCOPED_TRACE(testing::Message()
                 << "opcode=" << test.opcode << " mip_filter=" << +test.mip_filter
                 << " lod=" << test.lod << " min=" << test.minimum);
    cu_->write_sgpr(sb_ + 9, test.minimum | (0xfffu << 12));
    cu_->write_sgpr(sb_ + 10, uint32_t{test.mip_filter} << 26);
    if (test.clear_mode)
      wf_->set_mode_raw(wf_->mode_raw() & ~(1u << 10));
    wf_->set_exec(test.active ? 1 : 0);
    const unsigned status_reg = test.opcode == rdna3_5::kImageGather4LzMimg ? 12 : 9;
    for (unsigned lane = 0; lane < 4; ++lane) {
      write_coordinates(lane, {0, 0, std::bit_cast<uint32_t>(test.lod)});
      cu_->write_vgpr(vb_ + status_reg, lane, initial);
    }
    ASSERT_NO_FATAL_FAILURE(
        issue_image(test.opcode, {.dim = 1, .dmask = 1, .vdata = 8, .lwe = test.lwe, .ssamp = 2}));
    EXPECT_EQ(cu_->read_vgpr(vb_ + status_reg, 0), test.status);
    EXPECT_EQ(cu_->read_vgpr(vb_ + status_reg, 1), initial);
    ASSERT_TRUE(cu_->execute_instruction(read_mode.value().get(), *wf_).succeeded());
    EXPECT_EQ(cu_->read_sgpr(sb_ + 16), uint32_t{test.sticky});
  }
}

TEST_P(Gfx11ImageFeedbackTest, UnfilteredSamplesAndQueriesHonorTheFirstResidentMip) {
  Image image;
  image.width = image.height = 64;
  image.pitch = 1;
  image.mip_levels = 4;
  image.last_level = 3;
  image.rdna_format = 22;
  auto resource = rdna_descriptor(image, arch());
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  cu_->write_sgpr(sb_ + 8, 2 | (2 << 3));
  cu_->write_sgpr(sb_ + 9, 0xfffu << 12);
  cu_->write_sgpr(sb_ + 10, 0); // Disabled mip filtering.
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  // GFX11 linear R32 mip origins for a 64x64 four-level image.
  memory_.write32(kBase + 3584 * 4, 0x3f800000u);
  memory_.write32(kBase + 1536 * 4, 0x40000000u);
  memory_.write32(kBase + 512 * 4, 0x40400000u);
  for (uint32_t warning : {0u, 384u}) {
    for (uint32_t minimum : {0u, 384u, 512u, 1023u}) {
      cu_->write_sgpr(sb_ + 5, resource[5] | (warning << 8) | ((minimum & 31u) << 27));
      cu_->write_sgpr(sb_ + 6, minimum >> 5);
      for (auto op : {rdna3_5::kImageSampleLMimg, rdna3_5::kImageGather4LzMimg}) {
        for (uint8_t flags : {0u, 1u, 2u}) {
          SCOPED_TRACE(testing::Message() << "op=" << op << " min=" << minimum << " warning="
                                          << warning << " flags=" << unsigned(flags));
          const bool gather =
              op == rdna3_5::kImageGather4LMimg || op == rdna3_5::kImageGather4LzMimg;
          cu_->write_vgpr(vb_ + (gather ? 12 : 9), 0, 0xdeadbeef);
          ASSERT_NO_FATAL_FAILURE(issue_image(op, {.dim = 1,
                                                   .dmask = 1,
                                                   .vdata = 8,
                                                   .tfe = uint8_t(flags == 1),
                                                   .lwe = uint8_t(flags == 2),
                                                   .ssamp = 2}));
          const bool empty =
              minimum > 3 * 256 || (op == rdna3_5::kImageGather4LzMimg && minimum >= 256);
          const float result =
              empty ? 0.0f : float(minimum / 256 + (op != rdna3_5::kImageGetLodMimg));
          EXPECT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 0)), result);
          EXPECT_EQ(cu_->read_vgpr(vb_ + (gather ? 12 : 9), 0), 0xdeadbeefu);
          EXPECT_EQ(wf_->mode_raw() & (1u << 10), 0u);
        }
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, SparseAndResidentLanesCompleteOneImageLoad) {
  class SinglePage final : public amdgpu::AddressSpaceTranslator {
  public:
    bool published = true;
    std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
      static const auto holes = std::make_shared<const amdgpu::PrtRanges>(
          amdgpu::PrtRanges{{kBase + 4096, kBase + 8192}});
      return published ? holes : nullptr;
    }

  private:
    amdgpu::IdentityAddressSpaceTranslator identity_;
    amdgpu::VmTranslationResult translate(uint64_t address, size_t bytes,
                                          amdgpu::VmAccessKind access) const override {
      if ((address >> 12) != (kBase >> 12))
        return {};
      return identity_.translate(address, bytes, access);
    }
  };
  auto image = rdna_descriptor(Image{.width = 2048, .pitch = 2048}, arch());
  image[5] |= 256u << 8; // MIN_LOD_WARN = 1 for the LWE controls.
  for (uint32_t i = 0; i < image.size(); ++i)
    cu_->write_sgpr(sb_ + i, image[i]);
  memory_.write32(kBase, 0x12345678);
  write_coordinates(0, {0, 0, 0});
  write_coordinates(2, {1024, 0, 0}); // X=1024 begins the missing VA page.
  auto decoder = Decoder::create(arch());
  amdgpu::GpuVm vm;
  auto pages = std::make_shared<SinglePage>();
  uint32_t faults = 0;
  auto handle = vm.register_address_space(
      1, pages, std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_),
      [&](uint64_t, amdgpu::VmAccessKind) { ++faults; });
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  for (bool tfe : {false, true}) {
    for (uint32_t mask : {0u, 1u}) {
      SCOPED_TRACE(testing::Message() << "tfe=" << tfe << " mask=" << mask);
      fill_destination(0xdeadbeef);
      rdna3_5::MimgBuilderFields fields{};
      fields.dim = 1;
      fields.dmask = mask;
      fields.vdata = 8;
      fields.unorm = true;
      fields.tfe = tfe;
      const auto words = rdna3_5::build_mimg(rdna3_5::kImageLoadMimg, fields);
      auto decoded = decoder->decode(words.data());
      ASSERT_TRUE(decoded.succeeded());
      auto inst = std::move(decoded).value();
      inst->execute(*inst, wf_);
      ASSERT_FALSE(wf_->instruction_execution_failed());
      ASSERT_NE(inst->data(), nullptr);
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), mask ? 0x12345678u : 0u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeef);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 2), 0u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0xdeadbeef);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 2), mask && tfe ? 1u : 0xdeadbeef);
      EXPECT_EQ(faults, 0u);
    }
  }
  // The sampler requests LOD zero, below the resource's warning threshold.
  // A missing page takes precedence over LWE. DMASK=0 suppresses the fetch
  // and its NACK but still reports the requested LOD warning.
  cu_->write_sgpr(sb_ + 8, 2 | (2 << 3));
  cu_->write_sgpr(sb_ + 10, 1u << 26);
  write_coordinates(0, {0, 0, 0});
  write_coordinates(2, {std::bit_cast<uint32_t>(0.5f), 0, 0});
  for (uint32_t mask : {0u, 1u}) {
    fill_destination(0xdeadbeef);
    rdna3_5::MimgBuilderFields fields{};
    fields.dim = 1;
    fields.ssamp = 2;
    fields.dmask = mask;
    fields.vdata = 8;
    fields.tfe = fields.lwe = true;
    const auto words = rdna3_5::build_mimg(rdna3_5::kImageSampleLzMimg, fields);
    auto decoded = decoder->decode(words.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    inst->execute(*inst, wf_);
    ASSERT_FALSE(wf_->instruction_execution_failed());
    ASSERT_NE(inst->data(), nullptr);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), mask ? 0x12345678u : 0u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 2), 0u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 2u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 2), mask ? 1u : 2u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 1), 0xdeadbeefu);
    EXPECT_EQ(faults, 0u);
  }
  for (unsigned flags = 0; flags < 4; ++flags) {
    // A dropped D16 sample returns one zero half. Only LWE can produce status.
    fill_destination(0xdeadbeef);
    const bool tfe = flags & 1, lwe = flags & 2;
    const auto words = rdna3_5::build_mimg(
        rdna3_5::kImageSampleLzMimg,
        {.dim = 1, .dmask = 0, .d16 = 1, .vdata = 8, .tfe = tfe, .lwe = lwe, .ssamp = 2});
    auto decoded = decoder->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    ASSERT_NE(inst->data(), nullptr);
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    for (unsigned lane : {0u, 2u}) {
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), 0xdead0000u) << flags;
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, lane), lwe ? 2u : 0xdeadbeefu) << flags;
    }
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 1), 0xdeadbeefu);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 1), 0xdeadbeefu);
    EXPECT_EQ(faults, 0u);
  }
  // An ordinary unmapped page faults even when the shader requests TFE.
  pages->published = false;
  cu_->write_sgpr(sb_ + 5, image[5] & ~(0xfffu << 8));
  write_coordinates(2, {1024, 0, 0});
  rdna3_5::MimgBuilderFields ordinary{};
  ordinary.dim = 1;
  ordinary.dmask = 1;
  ordinary.vdata = 8;
  ordinary.unorm = ordinary.tfe = true;
  const auto words = rdna3_5::build_mimg(rdna3_5::kImageLoadMimg, ordinary);
  auto decoded = decoder->decode_window(words);
  ASSERT_TRUE(decoded.succeeded());
  auto load = std::move(decoded).value();
  ASSERT_TRUE(cu_->execute_instruction(load.get(), *wf_).succeeded());
  amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
  EXPECT_EQ(pipeline.issue(load.release(), *wf_), amdgpu::VmAccessOutcome::Faulted);
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(AllImageAccessTest, CompressedDescriptorsStopBeforeAccessingOrdinaryBacking) {
  for (uint32_t swizzle : {0u, arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2u : 22u}) {
    SCOPED_TRACE(swizzle);
    bind_image(2, 64, 64, swizzle);
    const uint32_t word6 = cu_->read_sgpr(sb_ + 6);
    for (uint32_t flag : {21u, 20u}) {
      if (flag == 20 && arch_is_cdna_4_or_lower(arch()))
        continue;
      cu_->write_sgpr(sb_ + 6, word6 | (1u << flag));
      for (bool load : {false, true}) {
        wf_->clear_instruction_execution_error();
        EXPECT_FALSE(decoded_issue(load, 2));
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
      }
    }
  }
}

TEST_P(AllImageAccessTest, UnsupportedLayoutIsDiagnosedEvenWithoutInBoundsLanes) {
  bind_image(2, 3, 2, 8);
  for (uint64_t lanes : {uint64_t{0}, uint64_t{1}}) {
    wf_->clear_instruction_execution_error();
    wf_->set_exec(lanes);
    write_coordinates(0, {100, 100, 100});
    EXPECT_FALSE(decoded_issue(true, 2));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(AllImageAccessTest, StoreRequiresUnnormalizedCoordinates) {
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "VIMAGE has no UNORM control";
  bind_image(2);
  unnormalized_ = false;
  EXPECT_FALSE(decoded_issue(false, 2));
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnsupportedOperandValue);
}

TEST_P(AllImageAccessTest, LegalCubeAndMsaaDimensionsStopAsUnimplemented) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 has no DIM field";
  bind_image(2);
  for (uint32_t dimension : {3u, 6u, 7u}) {
    SCOPED_TRACE(dimension);
    wf_->clear_instruction_execution_error();
    dimension_override_ = dimension;
    EXPECT_FALSE(decoded_issue(true, 2));
    const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
    EXPECT_EQ(wf_->instruction_execution_error(),
              gfx10 && dimension >= amdgpu::kImageDim2dMsaa
                  ? amdgpu::InstructionExecutionError::UnsupportedOperandValue
                  : amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(AllImageAccessTest, ScatteredAndPackedCoordinatesReachTheSameTexel) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 image coordinates are sequential";
  wf_->set_exec(1);
  bind_image(3);
  scattered_ = true;
  // Independent 3D linear layout: byte offset = slice + row + 2 DWORDs.
  const uint32_t byte_offset = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 648 : 776;
  for (bool packed : {false, true}) {
    packed_coordinates_ = packed;
    cu_->write_vgpr(vb_, 0, packed ? 2 | (1u << 16) : 2);
    cu_->write_vgpr(vb_ + 10, 0, 1);
    cu_->write_vgpr(vb_ + 12, 0, 1);
    cu_->write_vgpr(vb_ + 8, 0, 0x10203040u + packed);
    ASSERT_TRUE(decoded_issue(false, 3));
    EXPECT_EQ(memory_.read32(kBase + byte_offset), 0x10203040u + packed);
    const uint32_t load_value = 0xabcde000u + packed;
    memory_.write32(kBase + byte_offset, load_value);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    cu_->write_vgpr(vb_ + 8, 0, 0);
    ASSERT_TRUE(decoded_issue(true, 3));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), load_value);
  }
}

TEST_P(AllImageAccessTest, WaitChecksConsumeTheEncodedCoordinateRegisters) {
  bind_image(3);
  wf_->set_exec(1);
  scattered_ = !arch_is_cdna_4_or_lower(arch());
  auto instruction = decoded_instruction(false, 3);
  ASSERT_NE(instruction, nullptr);
  auto &state = wf_->ensure_memory_wait_scoreboard();
  auto check = [&](bool reads_coordinates) {
    for (uint16_t pending : {0, 1, 2, 3, 10, 12}) {
      SCOPED_TRACE(testing::Message()
                   << "v" << pending << " reads_coordinates=" << reads_coordinates);
      state.clear();
      const auto counter = amdgpu::WaitCounterKind::Load;
      state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, pending, 1}, counter, 0xf});
      unsigned reports = 0;
      state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      state.check_instruction(*instruction, *wf_);
      const bool encoded =
          scattered_ ? pending == 0 || pending == 10 || pending == 12 : pending < 3;
      EXPECT_EQ(reports, unsigned(reads_coordinates && encoded));
    }
  };
  check(true);
  if (amdgpu::has_gfx11_plus_image_operands(arch())) {
    // Keep the same NSA instruction and 3D resource, but invalidate its format.
    cu_->write_sgpr(sb_ + 1, 0);
    check(false);
  }
}

TEST_P(AllImageAccessTest, LivenessUsesScatteredCoordinatesAndScaledDescriptor) {
  scattered_ = !arch_is_cdna_4_or_lower(arch());
  resource_selector_ = 4;
  for (bool load : {false, true}) {
    for (bool packed : {false, true}) {
      SCOPED_TRACE(testing::Message() << "load=" << load << " packed=" << packed);
      packed_coordinates_ = packed;
      std::unique_ptr<Instruction> instruction = decoded_instruction(load, 3);
      ASSERT_NE(instruction, nullptr);
      const InstDefUse def_use(*instruction, wf_->wf_size());
      EXPECT_TRUE(def_use.uses.contains({RegClass::SGPR, 16, 8}));
      EXPECT_FALSE(def_use.uses.contains({RegClass::SGPR, 4, 1}));
      for (uint16_t reg : {0, 1, 2, 3, 10, 12}) {
        const bool used =
            reg == 0 || reg == (scattered_ ? 10 : 1) || (!packed && reg == (scattered_ ? 12 : 2));
        EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, reg, 1}), used) << reg;
      }
    }
  }
}

TEST_P(AllImageAccessTest, WaitsAndRegisterReadsUseTheBoundTransferShape) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "RDNA DIM can supply more coordinates than the resource requires";
  class Inputs final : public ExecutionPlugin {
  public:
    Inputs() : ExecutionPlugin("image_coordinate_inputs") {}
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t bytes) override {
      if (lanes & 1)
        read.at(reg - wave->vgpr_alloc().base) |= bytes;
    }
    std::array<uint8_t, 32> read{};
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Inputs>();
  auto *inputs = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  auto &state = wf_->ensure_memory_wait_scoreboard();
  wf_->set_exec(1);
  scattered_ = true;
  for (unsigned dimensions : {1, 2}) {
    bind_image(dimensions);
    for (bool packed : {false, true}) {
      packed_coordinates_ = packed;
      // A 2D-array instruction on a plain 2D resource omits the array layer.
      dimension_override_ = dimensions == 2 ? amdgpu::kImageDim2dArray : amdgpu::kImageDim1d;
      auto instruction = decoded_instruction(true, dimensions);
      ASSERT_NE(instruction, nullptr);
      inputs->read.fill(0);
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wf_).succeeded());
      for (uint16_t reg : {0, 10, 12}) {
        const unsigned expected = reg == 0 ? (packed && dimensions == 1 ? 3 : 15)
                                  : reg == 10 && !packed && dimensions == 2 ? 15
                                                                            : 0;
        EXPECT_EQ(inputs->read[reg], expected);
        for (uint8_t bytes : {3, 12}) {
          unsigned reports = 0;
          state.clear();
          const auto counter = amdgpu::WaitCounterKind::Load;
          state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, reg, 1}, counter, bytes});
          state.bind(0x200, &reports,
                     [](void *ctx, const auto &) { ++*static_cast<unsigned *>(ctx); });
          state.check_instruction(*instruction, *wf_);
          EXPECT_EQ(reports, unsigned(bool(expected & bytes))) << reg << ' ' << unsigned(bytes);
        }
      }
    }
  }
}

TEST_P(AllImageAccessTest, A16CoordinateWaitChecksIgnoreTheUnusedHighHalf) {
  bind_image(3);
  wf_->set_exec(1);
  scattered_ = !arch_is_cdna_4_or_lower(arch());
  packed_coordinates_ = true;
  std::unique_ptr<Instruction> instruction = decoded_instruction(false, 3);
  ASSERT_NE(instruction, nullptr);
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  for (uint8_t bytes : {0x3, 0xc}) {
    SCOPED_TRACE(bytes);
    state.clear();
    const auto counter = amdgpu::WaitCounterKind::Load;
    const uint16_t last = scattered_ ? 10 : 1;
    state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, last, 1}, counter, bytes});
    unsigned reports = 0;
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.check_instruction(*instruction, *wf_);
    EXPECT_EQ(reports, bytes == 0x3 ? 1u : 0u);
  }
}

TEST_P(AllImageAccessTest, D16LoadWaitOwnershipDistinguishesPreservedAndZeroedHighHalves) {
  wf_->set_exec(1);
  packed_data_ = true;
  std::unique_ptr<Instruction> instruction = decoded_instruction(true, 3);
  ASSERT_NE(instruction, nullptr);
  const InstDefUse def_use(*instruction, wf_->wf_size());
  EXPECT_EQ(def_use.uses.contains({RegClass::VGPR, 8, 1}), !cu_->sram_ecc());
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  for (uint8_t bytes : {0x3, 0xc}) {
    SCOPED_TRACE(bytes);
    state.clear();
    unsigned reports = 0;
    cu_->track_memory_wait(*instruction, *wf_);
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.access({RegClass::VGPR, 8, 1}, 1, bytes, false);
    EXPECT_EQ(reports, bytes == 0x3 || cu_->sram_ecc() ? 1u : 0u);
  }
}

TEST_P(AllImageAccessTest, TiledBaseXorStopsBeforeImageMemoryAccess) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  const bool gfx11 = arch() == ROCJITSU_CODE_ARCH_RDNA3 || arch() == ROCJITSU_CODE_ARCH_RDNA3_5;
  wf_->set_exec(1);
  for (uint32_t swizzle : {gfx12 ? 2u : 22u, gfx12 ? 3u : 27u}) {
    if (!gfx11 && !gfx12 && swizzle == 27)
      continue;
    bind_image(2, 64, 64, swizzle);
    cu_->write_sgpr(sb_, (kBase + 256) >> 8);
    for (bool tfe : {false, true}) {
      if (tfe && !gfx11 && !gfx12)
        continue;
      texel_fault_ = tfe;
      for (uint32_t x : {3u, 64u}) {
        write_coordinates(0, {x, 1, 0});
        for (bool load : {false, true}) {
          SCOPED_TRACE(testing::Message() << "swizzle=" << swizzle << " tfe=" << tfe << " x=" << x
                                          << " load=" << load);
          wf_->clear_instruction_execution_error();
          memory_.write32(kBase + 256 + 28, 0x12345678);
          cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
          cu_->write_vgpr(vb_ + 9, 0, 0x87654321);
          EXPECT_FALSE(decoded_issue(load, 2));
          EXPECT_EQ(wf_->instruction_execution_error(),
                    amdgpu::InstructionExecutionError::UnimplementedInstruction);
          EXPECT_EQ(memory_.read32(kBase + 256 + 28), 0x12345678u);
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
          EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x87654321u);
        }
      }
    }
  }
}

TEST_P(AllImageAccessTest, ThickDisplayVolumesStopBeforeImageMemoryAccess) {
  if (arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX9 display volumes are thin; GFX12 has different swizzle encodings";
  bind_image(3, 64, 64, 26);
  for (bool load : {false, true}) {
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(decoded_issue(load, 3));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
}

TEST_P(AllImageAccessTest, VolumeFeedbackUsesTheOrdinaryGfx11SwizzleLayout) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  wf_->set_exec(1);
  write_coordinates(0, {8, 0, 1});
  auto decoder = Decoder::create(arch());
  for (const uint32_t swizzle : {24, 25, 27, 29}) {
    bind_image(Image{
        .width = 64, .height = 64, .depth = 8, .type = amdgpu::kImage3d, .swizzle_mode = swizzle});
    for (const bool load : {false, true}) {
      for (const bool feedback : {false, true}) {
        SCOPED_TRACE(testing::Message() << swizzle << " load=" << load << " tfe=" << feedback);
        auto fields = this->fields<rdna3_5::MimgBuilderFields>(3);
        fields.tfe = feedback;
        auto decoded = decoder->decode_window(
            rdna3_5::build_mimg(load ? rdna3_5::kImageLoadMimg : rdna3_5::kImageStoreMimg, fields));
        ASSERT_TRUE(decoded.succeeded());
        auto instruction = std::move(decoded).value();
        const bool supported = swizzle == 24 || swizzle == 27;
        EXPECT_EQ(cu_->execute_instruction(instruction.get(), *wf_).succeeded(), supported);
        EXPECT_EQ(instruction->data() != nullptr, supported);
        if (!supported) {
          EXPECT_EQ(wf_->instruction_execution_error(),
                    amdgpu::InstructionExecutionError::UnimplementedInstruction);
        }
      }
    }
  }
}

TEST_P(AllImageAccessTest, Gfx11FeedbackVolumesKeepDccAddressAndSlice) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 || cu_->image_address_config().gb_addr_config != 0x545)
    GTEST_SKIP();
  bind_image(
      Image{.width = 64, .height = 64, .depth = 8, .type = amdgpu::kImage3d, .swizzle_mode = 27});
  cu_->write_sgpr(sb_ + 6, cu_->read_sgpr(sb_ + 6) | (1u << 21) | (1u << 19));
  cu_->write_sgpr(sb_ + 7, 0x20);
  wf_->set_exec(1);
  write_coordinates(0, {8, 0, 1});
  auto decoder = Decoder::create(arch());
  auto encoding = fields<rdna3_5::MimgBuilderFields>(3);
  encoding.tfe = 1;
  auto decoded = decoder->decode_window(rdna3_5::build_mimg(rdna3_5::kImageLoadMimg, encoding));
  ASSERT_TRUE(decoded.succeeded());
  auto instruction = std::move(decoded).value();
  ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wf_).succeeded());
  const auto *state = instruction->data_as<amdgpu::VectorMemState>();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->per_lane_addr[0], 0x22400u);
  ASSERT_TRUE(state->image_metadata);
  EXPECT_EQ(state->image_metadata->metadata, 0x200000u);
  EXPECT_EQ(state->image_metadata->layers[0], 1u);
}

TEST(ImageOpcodeBuilderTest, SplitImageOpcodeAndRawBufferModifiersRoundTrip) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA2);
  const auto check = [&](const auto &words, const char *mnemonic) {
    auto instruction = decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    EXPECT_EQ(instruction.value()->mnemonic(), mnemonic);
  };
  check(rdna2::build_mimg(rdna2::kImageMsaaLoadMimg), "image_msaa_load");
  check(rdna2::build_mimg(rdna2::kImageLoadMimg), "image_load");
  check(rdna2::build_mubuf(rdna2::kBufferLoadFormatD16XMubuf, {.opm = 1}),
        "buffer_load_format_d16_x");
  check(rdna2::build_mubuf(rdna2::kBufferLoadFormatXMubuf), "buffer_load_format_x");
  check(rdna2::build_mtbuf(rdna2::kTbufferLoadFormatD16XMtbuf, {.opm = 1}),
        "tbuffer_load_format_d16_x");
  check(rdna2::build_mtbuf(rdna2::kTbufferLoadFormatXMtbuf), "tbuffer_load_format_x");
}

INSTANTIATE_TEST_SUITE_P(Architectures, AllImageAccessTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

enum class SampleTestOperation { Zero, Lod, Gradients, Gather, Offset };

// Raw S# field encodings independent of the sampler decoder under test.
enum class Clamp : uint32_t { Repeat = 0, Mirror = 1, Edge = 2, MirrorOnce = 3, Border = 6 };
enum class MipFilter : uint32_t { None = 0, Point = 1, Linear = 2 };

struct Sampler {
  Clamp clamp_x = Clamp::Edge, clamp_y = Clamp::Edge, clamp_z = Clamp::Edge;
  bool linear_xy = false; // Program the same XY_MAG_FILTER and XY_MIN_FILTER.
  MipFilter mip = MipFilter::None;
  uint32_t min_lod_fixed8 = 0, max_lod_fixed8 = 3 * 256;
  int32_t bias_fixed8 = 0;
  bool normalized = true;
  uint32_t perf_mip = 0, perf_z = 0;
};

class ImageSamplerControlTest : public AllImageAccessTest {
protected:
  DecodeResult modern_sample_lz(bool unrm, uint32_t mask, bool tfe,
                                uint8_t dim = amdgpu::kImageDim2d, bool r128 = false) {
    rdna4::VsampleBuilderFields gfx12_fields{};
    gfx12_fields.dim = dim;
    gfx12_fields.r128 = r128;
    gfx12_fields.dmask = mask;
    gfx12_fields.unorm = unrm;
    gfx12_fields.tfe = tfe;
    gfx12_fields.vdata = gfx12_fields.samp = 8;
    gfx12_fields.vaddr1 = 1;
    gfx12_fields.vaddr2 = 2;
    rdna3_5::MimgBuilderFields gfx11_fields{};
    gfx11_fields.dim = dim;
    gfx11_fields.r128 = r128;
    gfx11_fields.dmask = mask;
    gfx11_fields.unorm = unrm;
    gfx11_fields.tfe = tfe;
    gfx11_fields.vdata = 8;
    gfx11_fields.ssamp = 2;
    auto decoder = Decoder::create(arch());
    return arch() == ROCJITSU_CODE_ARCH_RDNA4
               ? decoder->decode_window(
                     rdna4::build_vsample(rdna4::kImageSampleLzVsample, gfx12_fields))
               : decoder->decode_window(
                     rdna3_5::build_mimg(rdna3_5::kImageSampleLzMimg, gfx11_fields));
  }

  void issue_modern_sample_lz(bool unrm, uint32_t mask, bool tfe = false,
                              uint8_t dim = amdgpu::kImageDim2d) {
    auto instruction = modern_sample_lz(unrm, mask, tfe, dim);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_TRUE(cu_->execute_instruction(instruction.value().get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(instruction.value().release(), *wf_),
              amdgpu::VmAccessOutcome::Complete);
  }

  void bind_sample_image(Image image, bool floating = true) {
    image.num_format = floating ? 7 : kNfmtUint;
    image.rdna_format = floating ? 22 : kR32Uint;
    bind_image(image);
  }

  void bind_sampler(const Sampler &sampler = {}) {
    const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
    const uint32_t xy_filter = sampler.linear_xy ? 1u : 0u;
    // Z_FILTER_NONE, zero anisotropy, transparent black border.
    const std::array<uint32_t, 4> words{
        static_cast<uint32_t>(sampler.clamp_x) | (static_cast<uint32_t>(sampler.clamp_y) << 3) |
            (static_cast<uint32_t>(sampler.clamp_z) << 6) | (sampler.normalized ? 0u : 1u << 15),
        sampler.min_lod_fixed8 | (sampler.max_lod_fixed8 << (gfx12 ? 13 : 12)) |
            (gfx12 ? 0u : sampler.perf_mip << 24) | (sampler.perf_z << 28),
        (static_cast<uint32_t>(sampler.bias_fixed8) & 0x3fffu) | (xy_filter << 20) |
            (xy_filter << 22) | (static_cast<uint32_t>(sampler.mip) << 26) |
            (gfx12 ? (sampler.perf_mip & 3) << 30 : 0u),
        gfx12 ? sampler.perf_mip >> 2 : 0u};
    for (uint32_t word = 0; word < words.size(); ++word)
      cu_->write_sgpr(sb_ + 8 + word, words[word]);
  }

  template <typename Fields, typename Builder>
  std::array<uint32_t, 3>
  sample_control_mimg_words(Builder builder, std::array<uint16_t, 5> opcodes,
                            SampleTestOperation operation, bool d16, uint32_t sampler) {
    return sample_mimg_words<Fields>(builder, opcodes[static_cast<size_t>(operation)], d16,
                                     sampler);
  }

  std::unique_ptr<Instruction> sample_instruction(SampleTestOperation operation, bool d16 = false,
                                                  uint32_t sampler = 8) {
    std::array<uint32_t, 3> words{};
    switch (arch()) {
    case ROCJITSU_CODE_ARCH_CDNA1:
      words = sample_control_mimg_words<cdna1::MimgBuilderFields>(
          cdna1::build_mimg,
          {cdna1::kImageSampleLzMimg, cdna1::kImageSampleLMimg, cdna1::kImageSampleDMimg,
           cdna1::kImageGather4LzMimg, cdna1::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_RDNA1:
      words = sample_control_mimg_words<rdna1::MimgBuilderFields>(
          rdna1::build_mimg,
          {rdna1::kImageSampleLzMimg, rdna1::kImageSampleLMimg, rdna1::kImageSampleDMimg,
           rdna1::kImageGather4LzMimg, rdna1::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_RDNA2:
      words = sample_control_mimg_words<rdna2::MimgBuilderFields>(
          rdna2::build_mimg,
          {rdna2::kImageSampleLzMimg, rdna2::kImageSampleLMimg, rdna2::kImageSampleDMimg,
           rdna2::kImageGather4LzMimg, rdna2::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_RDNA3:
      words = sample_control_mimg_words<rdna3::MimgBuilderFields>(
          rdna3::build_mimg,
          {rdna3::kImageSampleLzMimg, rdna3::kImageSampleLMimg, rdna3::kImageSampleDMimg,
           rdna3::kImageGather4LzMimg, rdna3::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_RDNA3_5:
      words = sample_control_mimg_words<rdna3_5::MimgBuilderFields>(
          rdna3_5::build_mimg,
          {rdna3_5::kImageSampleLzMimg, rdna3_5::kImageSampleLMimg, rdna3_5::kImageSampleDMimg,
           rdna3_5::kImageGather4LzMimg, rdna3_5::kImageSampleOMimg},
          operation, d16, sampler);
      break;
    case ROCJITSU_CODE_ARCH_CDNA2: {
      // CDNA2's public ISA input exposes only base IMAGE_SAMPLE. A single-level
      // sample without mip filtering has the same controls as the zero form.
      if (operation != SampleTestOperation::Zero)
        return {};
      words = sample_mimg_words<cdna2::MimgBuilderFields>(cdna2::build_mimg,
                                                          cdna2::kImageSampleMimg, d16, sampler);
      break;
    }
    case ROCJITSU_CODE_ARCH_RDNA4: {
      auto encoding = rdna4::VsampleBuilderFields{};
      encoding.dim = dimension_override_.value_or(1);
      encoding.dmask = 1;
      encoding.vdata = 8;
      encoding.rsrc = 0;
      encoding.samp = sampler;
      encoding.tfe = texel_fault_;
      encoding.vaddr0 = 0;
      encoding.vaddr1 = 1;
      encoding.vaddr2 = 2;
      encoding.vaddr3 = 3;
      encoding.d16 = d16;
      const uint16_t opcode =
          operation == SampleTestOperation::Zero        ? rdna4::kImageSampleLzVsample
          : operation == SampleTestOperation::Lod       ? rdna4::kImageSampleLVsample
          : operation == SampleTestOperation::Gradients ? rdna4::kImageSampleDVsample
          : operation == SampleTestOperation::Gather    ? rdna4::kImageGather4LzVsample
                                                        : rdna4::kImageSampleOVsample;
      words = rdna4::build_vsample(opcode, encoding);
      break;
    }
    default:
      return {};
    }
    std::unique_ptr<Decoder> decoder = Decoder::create(arch());
    DecodeResult decoded = decoder->decode(words.data());
    EXPECT_TRUE(decoded.succeeded());
    return decoded.succeeded() ? std::move(decoded).value() : nullptr;
  }

  bool sample(SampleTestOperation operation, bool d16 = false, uint32_t sampler = 8) {
    std::unique_ptr<Instruction> instruction = sample_instruction(operation, d16, sampler);
    if (!instruction)
      return false;
    instruction->execute(*instruction, wf_);
    if (!instruction->data())
      return false;
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    EXPECT_EQ(pipeline.issue(instruction.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    // Later cases seed backing storage directly, as in the base image fixture.
    cu_->l1_vector().flush_all();
    l2_.flush_all();
    return true;
  }
};

TEST_P(ImageSamplerControlTest, WaitChecksFollowDescriptorLodAndGradientOperands) {
  const bool gfx9 = arch() == ROCJITSU_CODE_ARCH_CDNA1;
  if (!gfx9 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "Covers descriptor-selected GFX9 tuples and GFX12 VSAMPLE selectors";
  wf_->set_exec(1);
  std::unique_ptr<Decoder> decoder = Decoder::create(arch());
  amdgpu::MemoryWaitScoreboard &state = wf_->ensure_memory_wait_scoreboard();
  const auto has_hazard = [&](const Instruction &instruction, uint16_t reg, uint8_t bytes) {
    state.clear();
    const auto counter = amdgpu::WaitCounterKind::Load;
    state.add({state.issue(counter), 0x100, 1, {RegClass::VGPR, reg, 1}, counter, bytes});
    unsigned reports = 0;
    state.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
    state.check_instruction(instruction, *wf_);
    return reports != 0;
  };
  for (uint32_t dimensions : {2u, 3u}) {
    bind_image(dimensions);
    for (bool packed : {false, true}) {
      SCOPED_TRACE(testing::Message() << dimensions << ':' << packed);
      if (gfx9) {
        packed_coordinates_ = packed;
        auto encoding = fields<cdna1::MimgBuilderFields>(dimensions);
        encoding.vdata = 28;
        encoding.ssamp = 2;
        auto words = image_instruction_words(cdna1::build_mimg, cdna1::kImageSampleLMimg, encoding);
        DecodeResult lod = decoder->decode(words.data());
        ASSERT_TRUE(lod.succeeded());
        const uint16_t lod_reg = packed ? dimensions / 2 : dimensions;
        EXPECT_TRUE(has_hazard(*lod.value(), lod_reg, 0x3));
        EXPECT_EQ(has_hazard(*lod.value(), lod_reg, 0xc), !packed || dimensions == 3);
        EXPECT_FALSE(has_hazard(*lod.value(), lod_reg + 1, 0xf));
        // Packed GFX9 gradients are excluded from execution. Score the
        // supported full-width gradient prefix and following coordinates.
        if (!packed) {
          words = image_instruction_words(cdna1::build_mimg, cdna1::kImageSampleDMimg, encoding);
          DecodeResult grad = decoder->decode(words.data());
          ASSERT_TRUE(grad.succeeded());
          EXPECT_TRUE(has_hazard(*grad.value(), 2 * dimensions - 1, 0xc));
          EXPECT_TRUE(has_hazard(*grad.value(), 3 * dimensions - 1, 0xc));
          EXPECT_FALSE(has_hazard(*grad.value(), 3 * dimensions, 0xf));
        }
      } else {
        rdna4::VsampleBuilderFields encoding{};
        encoding.dim = dimensions - 1;
        encoding.dmask = 1;
        encoding.vdata = 28;
        encoding.samp = 8;
        encoding.a16 = packed;
        encoding.vaddr0 = 1;
        encoding.vaddr1 = 12;
        encoding.vaddr2 = 14;
        encoding.vaddr3 = 20;
        const auto words = rdna4::build_vsample(rdna4::kImageSampleDVsample, encoding);
        DecodeResult grad = decoder->decode(words.data());
        ASSERT_TRUE(grad.succeeded());
        for (uint16_t reg : {1, 12, 14, 20})
          EXPECT_TRUE(has_hazard(*grad.value(), reg, 0xc)) << reg;
        EXPECT_FALSE(has_hazard(*grad.value(), 2, 0xf));
        const uint16_t first_coordinate = 21 + (dimensions == 3 ? 2 : 0);
        if (dimensions == 3) {
          EXPECT_TRUE(has_hazard(*grad.value(), first_coordinate - 1, 0xc));
        }
        const uint16_t last = first_coordinate + (packed ? (dimensions - 1) / 2 : dimensions - 1);
        EXPECT_TRUE(has_hazard(*grad.value(), last, 0x3));
        EXPECT_EQ(has_hazard(*grad.value(), last, 0xc), !packed || dimensions == 2);
        EXPECT_FALSE(has_hazard(*grad.value(), last + 1, 0xf));
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, UnnormalizedWrapUsesClampAndMirrorOnce) {
  Image image;
  image.width = 16;
  bind_sample_image(image);
  for (uint32_t x = 0; x < 16; ++x)
    memory_.write32(kBase + x * 4, std::bit_cast<uint32_t>(float(x)));
  wf_->set_exec(1);
  for (Clamp wrap : {Clamp::Repeat, Clamp::Mirror})
    for (bool linear : {false, true}) {
      bind_sampler({.clamp_x = wrap,
                    .clamp_y = wrap,
                    .clamp_z = Clamp::Repeat,
                    .linear_xy = linear,
                    .normalized = false});
      for (const auto &[x, mirror] :
           {std::pair{-INFINITY, 15.0f}, std::pair{-16.5f, 15.0f}, std::pair{-1.5f, 1.0f},
            std::pair{-0.5f, 0.0f}, std::pair{0.0f, 0.0f}, std::pair{16.5f, 15.0f}}) {
        cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(x));
        cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
        ASSERT_TRUE(sample(SampleTestOperation::Zero));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(wrap == Clamp::Mirror ? mirror
                                                                      : x < 1               ? 0.0f
                                                                              : 15.0f));
      }
    }
}

TEST_P(ImageSamplerControlTest, NativePitchSelectsTheHardwareRowAndFp16Filter) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "GFX11 and GFX12 native texture layout";

  wf_->set_exec(1);
  const auto bind_texture = [&](uint32_t format, uint32_t width, uint32_t pitch, bool mip = false,
                                uint32_t swizzle = 0) {
    const std::array<uint32_t, 8> words{
        uint32_t(kBase >> 8),
        (format << (gfx12 ? 17 : 20)) | ((width - 1) << 30) | (uint32_t(mip) << (gfx12 ? 12 : 16)),
        1u << 14,
        (9u << 28) | (swizzle << 20) | 4u | (uint32_t(mip) << (gfx12 ? 15 : 16)),
        pitch ? pitch - 1 : 0,
        0,
        0,
        0};
    for (unsigned i = 0; i < words.size(); ++i)
      cu_->write_sgpr(sb_ + i, words[i]);
  };

  // Physical gfx1100/gfx1201 samples of a 2x2 sRGB texture use 256/128-byte rows.
  bind_texture(66, 2, 0);
  bind_sampler();
  memory_.write32(kBase, 0xff000000u);
  memory_.write32(kBase + (gfx12 ? 0x84 : 0x104), 0x804080ffu);
  cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(2.0f));
  cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(2.0f));
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(1.0f));

  // The same filtering precision applies with an explicit pitch and for one-column views.
  // Clamping X to the first column reproduces the two-texel physical witness.
  const uint32_t quantum = (gfx12 ? 128u : 256u) / sizeof(uint16_t);
  bind_sampler({.linear_xy = true});
  for (uint32_t width : {1u, 2u}) {
    for (uint32_t pitch : {0u, 2 * quantum}) {
      for (bool mip : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "width=" << width << " pitch=" << pitch << " mip=" << mip);
        bind_texture(13, width, pitch, mip);
        const auto layout = amdgpu::image_mip_layout(gfx12, 0, 2, width, 2, mip ? 2 : 1, 0);
        ASSERT_TRUE(layout);
        const uint64_t base = kBase + layout->offset;
        memory_.write32(base, 0x3c000000u);
        memory_.write32(base + quantum * 2, 0);
        memory_.write32(base + 2 * quantum * 2, 0);
        memory_.write32(base + (pitch ? pitch : quantum) * sizeof(uint16_t), 0x34013bffu);
        cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(width == 1 ? 0.5f : 0.375f));
        cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.375f));
        cu_->write_vgpr(vb_ + 2, 0, 0);
        ASSERT_TRUE(sample(SampleTestOperation::Lod));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), width == 1 ? 0x3e7fe000u : 0x3ec7f400u);
      }
    }
  }
  // Physical gfx1100/gfx1201 return the same resident R16_FLOAT payload in all four cases.
  for (uint32_t swizzle : {0u, gfx12 ? 3u : 27u}) {
    bind_texture(13, 2, 0, false, swizzle);
    bind_sampler({.linear_xy = true});
    memory_.write32(kBase, 0x3c000000u);
    memory_.write32(kBase + (swizzle ? 4 : 2 * quantum), 0x34013bffu);
    write_coordinates(0, {std::bit_cast<uint32_t>(0.375f), std::bit_cast<uint32_t>(0.375f), 0});
    for (bool tfe : {false, true}) {
      SCOPED_TRACE(testing::Message() << "swizzle=" << swizzle << " tfe=" << tfe);
      cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeefu);
      cu_->write_vgpr(vb_ + 9, 0, 0);
      ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 1, tfe));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x3ec7f400u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0u);
    }
    // On both devices mirror-once maps normalized -0.5 to texel zero and +0.5 to one.
    bind_sampler({.clamp_x = Clamp::MirrorOnce});
    for (float x : {-0.5f, 0.5f}) {
      write_coordinates(0, {std::bit_cast<uint32_t>(x), std::bit_cast<uint32_t>(0.25f), 0});
      for (bool tfe : {false, true}) {
        cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
        ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 1, tfe));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), x < 0 ? 0u : 0x3f800000u);
      }
    }
    cu_->l1_vector().flush_all();
    l2_.flush_all();
  }
}

TEST_P(ImageSamplerControlTest, ForcedUnnormalizedSamplerCoordinatesClampLikeUnrm) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  Image image;
  image.width = 16;
  image.pitch = gfx12 ? 64 : 128;
  image.rdna_format = 13; // R16_FLOAT
  bind_image(image);
  memory_.write32(kBase, 0);
  memory_.write32(kBase + 28, 0x3c000000); // Texel 15 is 1.0.
  wf_->set_exec(1);
  for (bool use_instruction_unrm : {false, true}) {
    for (bool force_unnormalized : {false, true}) {
      SCOPED_TRACE(testing::Message() << "instruction unrm=" << use_instruction_unrm
                                      << " sampler force=" << force_unnormalized);
      bind_sampler(
          {.clamp_x = Clamp::Repeat, .clamp_y = Clamp::Edge, .normalized = !force_unnormalized});
      write_coordinates(0, {std::bit_cast<uint32_t>(16.5f), std::bit_cast<uint32_t>(0.5f), 0});
      ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(use_instruction_unrm, 1));
      EXPECT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 0)),
                use_instruction_unrm || force_unnormalized ? 1.0f : 0.0f);
    }
  }
}

TEST_P(ImageSamplerControlTest, CapturesRawTransferAndAtomicPayloadsBeforeGeneration) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  wf_->set_exec(1);
  write_coordinates(0, {1, 0, 0});
  const auto coords = amdgpu::contiguous_image_address_registers(0);
  bind_image(Image{.width = 4, .pitch = 256, .rdna_format = 5}); // R8_UINT at (1,0).
  for (bool signed_result : {false, true}) {
    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.is_load = true;
    const auto mode =
        signed_result ? amdgpu::ImageTransferMode::PackedSigned : amdgpu::ImageTransferMode::Packed;
    ASSERT_TRUE(amdgpu::prepare_image_transfer(*wf_, state, 0, 8, coords,
                                               {.dmask = 1, .dim = 1, .unorm = true}, ~0u,
                                               amdgpu::ImageSampleMode::Implicit, nullptr, mode)
                    .succeeded());
    EXPECT_EQ(state.per_lane_addr[0], kBase + 1);
    EXPECT_EQ(state.decoded_buffer_format.widths[0], 8u);
    EXPECT_EQ(state.decoded_buffer_format.number,
              signed_result ? amdgpu::BufferNumberFormat::Sint : amdgpu::BufferNumberFormat::Uint);
  }
  bind_image(Image{.width = 4, .rdna_format = kR32Uint});
  cu_->write_vgpr(vb_ + 8, 0, 99);
  cu_->write_vgpr(vb_ + 9, 0, 42);
  amdgpu::VectorMemState atomic(amdgpu::GLOBAL_MEM);
  atomic.atomic_op = amdgpu::AtomicOp::CMPSWAP;
  atomic.is_load = true;
  ASSERT_TRUE(amdgpu::prepare_image_transfer(*wf_, atomic, 0, 8, coords,
                                             {.dmask = 3, .dim = 1, .unorm = true})
                  .succeeded());
  EXPECT_EQ(atomic.per_lane_addr[0], kBase + 4);
  std::array<uint32_t, 2> captured{};
  ASSERT_GE(atomic.store_data.size(), sizeof(captured));
  std::memcpy(captured.data(), atomic.store_data.data(), sizeof(captured));
  EXPECT_EQ(captured, (std::array<uint32_t, 2>{99, 42}));
}

TEST_P(ImageSamplerControlTest, MultisampleFragmentAddressesUseTheCapturedDeviceConfiguration) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  Image image;
  image.width = image.height = image.pitch = 32;
  image.type = amdgpu::kImage2dMsaa;
  image.swizzle_mode = 27;
  image.last_level = 3;
  image.mip_levels = 4;
  image.rdna_format = kGfx11Rgba8Uint;
  bind_image(image);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 4});
  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.is_load = true;
  const auto issued = amdgpu::prepare_image_transfer(
      *wf_, state, 0, 8, amdgpu::contiguous_image_address_registers(0),
      {.dmask = 2, .dim = 6, .unorm = true}, ~0u, amdgpu::ImageSampleMode::Implicit, nullptr,
      amdgpu::ImageTransferMode::Msaa);
  if (cu_->image_address_config().gb_addr_config != 0x545) {
    EXPECT_TRUE(issued.failed());
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnsupportedOperandValue);
    return;
  }
  ASSERT_TRUE(issued.succeeded());
  ASSERT_TRUE(state.graphics_sample);
  constexpr std::array<uint64_t, 4> expected{kBase + 16, kBase + 20, kBase + 24, kBase + 28};
  for (unsigned tap = 0; tap < expected.size(); ++tap)
    EXPECT_EQ(state.graphics_sample->taps[tap].addresses[0], expected[tap]);
}

TEST_P(ImageSamplerControlTest, SampleOperandMasksMatchPreparation) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP();
  const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
  auto decoder = Decoder::create(arch());
  const auto decode = [&](uint8_t mask, bool bias, bool a16) {
    if (gfx10)
      return decoder->decode_window(rdna2::build_mimg(
          bias ? rdna2::kImageSampleBOMimg : rdna2::kImageSampleMimg,
          {.dim = 1, .dmask = mask, .vdata = 8, .ssamp = 2, .a16 = a16, .d16 = 1}));
    if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
      return decoder->decode_window(rdna4::build_vsample(
          bias ? rdna4::kImageSampleBOVsample : rdna4::kImageSampleVsample, {.dim = 1,
                                                                             .d16 = 1,
                                                                             .a16 = a16,
                                                                             .dmask = mask,
                                                                             .vdata = 8,
                                                                             .samp = 8,
                                                                             .vaddr1 = 1,
                                                                             .vaddr2 = 2,
                                                                             .vaddr3 = 3}));
    return decoder->decode_window(rdna3_5::build_mimg(
        bias ? rdna3_5::kImageSampleBOMimg : rdna3_5::kImageSampleMimg,
        {.dim = 1, .dmask = mask, .a16 = a16, .d16 = 1, .vdata = 8, .ssamp = 2}));
  };

  wf_->set_exec(1);
  bind_sample_image(Image{.width = 64, .height = 64, .pitch = 64});
  bind_sampler();
  memory_.write32(kBase, 0x3f800000u);
  for (unsigned lane = 0; lane < 4; ++lane)
    write_coordinates(lane, {0, 0, 0});

  // Plain samples issue, while offset/bias samples currently only expose decoded operands.
  // The newer TA overrides an empty fetch mask to one zero D16 component. GFX10 rejects it.
  for (uint8_t mask : {0, 1})
    for (bool bias : {false, true}) {
      auto decoded = decode(mask, bias, true);
      ASSERT_TRUE(decoded.succeeded());
      amdgpu::RegisterModifiers modifiers;
      decoded.value()->amdgpu_register_modifiers(modifiers);
      const bool partial = mask || !gfx10;
      EXPECT_EQ(modifiers.memory_result_last_bytes, partial ? 3u : 15u);
      const InstDefUse use(*decoded.value(), wf_->wf_size());
      EXPECT_EQ(use.uses.contains({RegClass::VGPR, 8, 1}), partial);
      if (bias)
        continue;
      cu_->write_vgpr(vb_ + 8, 0, 0xdeadc0de);
      wf_->clear_instruction_execution_error();
      const auto issued = cu_->execute_instruction(decoded.value().get(), *wf_);
      if (gfx10 && !mask) {
        ASSERT_TRUE(issued.failed());
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnsupportedOperandValue);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xdeadc0deu);
        wf_->clear_instruction_execution_error();
      } else {
        ASSERT_TRUE(issued.succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(decoded.value().release(), *wf_),
                  amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), mask ? 0xdead3c00u : 0xdead0000u);
      }
    }

  class PrefixReads final : public ExecutionPlugin {
  public:
    PrefixReads() : ExecutionPlugin("sample_prefix_reads") {}
    std::array<uint8_t, 2> bytes{};
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t mask) override {
      const uint32_t index = reg - wave->vgpr_alloc().base;
      if ((lanes & 1) && index < bytes.size())
        bytes[index] |= mask;
    }
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<PrefixReads>();
  auto *reads = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  wf_->set_exec(15); // All four lanes provide the GFX10 implicit derivatives.
  bind_sample_image(
      Image{.width = 64, .height = 64, .pitch = 64, .mip_levels = 2, .last_level = 1});
  bind_sampler({.mip = MipFilter::Point, .max_lod_fixed8 = 256});
  for (bool a16 : {false, true}) {
    auto decoded = decode(1, true, a16);
    ASSERT_TRUE(decoded.succeeded());
    amdgpu::RegisterModifiers modifiers;
    decoded.value()->amdgpu_register_modifiers(modifiers);
    ASSERT_EQ(modifiers.image_prefix_words, 2u);
    EXPECT_EQ(modifiers.image_prefix_low16_words, a16 ? 2u : 0u);
    const amdgpu::ImageFields fields{
        .dmask = 1, .dim = 1, .a16 = a16, .d16 = true, .offset = true, .lod_bias = true};
    // W7900 and R9700 give the same mip for 0x00003c00, 0x40003c00 and 0xc0003c00.
    // With an A16 bias of zero they select the other mip.
    for (unsigned lane = 0; lane < 4; ++lane) {
      const float x = (lane & 1) / 64.f, y = (lane >> 1) / 64.f;
      const uint32_t xy16 = (lane & 1 ? 0x2400u : 0u) | (lane & 2 ? 0x24000000u : 0u);
      write_coordinates(
          lane, {0, a16 ? 0xc0003c00u : 0x3f800000u, a16 ? xy16 : std::bit_cast<uint32_t>(x)});
      cu_->write_vgpr(vb_ + 3, lane, std::bit_cast<uint32_t>(y));
    }
    for (bool graphics : {false, true}) {
      if (graphics && gfx10)
        continue; // GFX10 uses the compute preparation.
      amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
      state.is_load = true;
      state.dst_reg_base = vb_ + 8;
      reads->bytes = {};
      auto coords = amdgpu::contiguous_image_address_registers(0);
      auto result = graphics ? amdgpu::prepare_image_transfer(*wf_, state, 0, 8, coords, fields, 8,
                                                              amdgpu::ImageSampleMode::Bias)
                             : amdgpu::prepare_image_sample(*wf_, state, 0, 8, coords, fields,
                                                            amdgpu::ImageSampleMode::Bias);
      ASSERT_TRUE(result.succeeded()) << graphics << '/' << a16;
      EXPECT_EQ(reads->bytes[0], 15u);
      EXPECT_EQ(reads->bytes[1], a16 ? 3u : 15u);
    }
  }
}

TEST_P(ImageSamplerControlTest, GraphicsPrefixesSelectMipTexelsAndMatchSourceWaits) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  class Reads final : public ExecutionPlugin {
  public:
    Reads() : ExecutionPlugin("graphics_prefix_reads") {}
    std::array<std::array<uint8_t, 5>, 4> bytes{};
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t mask) override {
      const unsigned index = reg - wave->vgpr_alloc().base;
      if (index < bytes[0].size())
        for (size_t lane = 0; lane < bytes.size(); ++lane)
          if (lanes & (1u << lane))
            bytes[lane][index] |= mask;
    }
  };
  // Carry the extended form through the scoreboard until its opcode is generated.
  class SampleInputs final : public Instruction {
  public:
    explicit SampleInputs(const Instruction &sample) : Instruction("image_sample", nullptr) {
      sample.amdgpu_register_modifiers(inputs);
    }
    amdgpu::RegisterModifiers inputs;
    void amdgpu_register_modifiers(amdgpu::RegisterModifiers &m) const override { m = inputs; }
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Reads>();
  auto *reads = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  wf_->set_exec(1);
  const auto coords = amdgpu::contiguous_image_address_registers(0);
  const auto sample = sample_instruction(SampleTestOperation::Zero);
  ASSERT_TRUE(sample);
  for (bool gradient : {true, false}) {
    const auto dim = gradient ? amdgpu::kImageDim1d : amdgpu::kImageDim2d;
    const uint32_t height = gradient ? 1 : 64;
    bind_sample_image(Image{.width = 64,
                            .height = height,
                            .pitch = 1,
                            .type = gradient ? amdgpu::kImage1d : amdgpu::kImage2d,
                            .mip_levels = 3,
                            .last_level = 2});
    bind_sampler({.mip = MipFilter::Point, .max_lod_fixed8 = 512});
    cu_->write_sgpr(sb_ + 8, cu_->read_sgpr(sb_ + 8) | (1u << 12)); // LESS
    const auto mode =
        gradient ? amdgpu::ImageSampleMode::Derivatives16 : amdgpu::ImageSampleMode::Bias;
    const amdgpu::ImageFields fields{.dmask = 1,
                                     .dim = dim,
                                     .a16 = true,
                                     .tfe = true,
                                     .gather = !gradient,
                                     .gradient16 = gradient,
                                     .offset = true,
                                     .lod_bias = !gradient,
                                     .compare = !gradient,
                                     .horizontal = !gradient};
    cu_->write_vgpr(vb_, 0, 1);                                        // Signed X offset.
    cu_->write_vgpr(vb_ + 1, 0, gradient ? 0xabcd2c00u : 0xdead3c00u); // dx = 1/16 or bias = 1.
    cu_->write_vgpr(vb_ + 2, 0, gradient ? 0xdead0000u : std::bit_cast<uint32_t>(3.f));
    for (unsigned lane = 0; lane < 4; ++lane) {
      // 1D X = 0.5; 2D quad at (0.25,0.25) with one base-level texel per lane.
      const uint32_t uv = 0x34003400u + ((lane & 1) ? 0x40u : 0) + ((lane & 2) ? 0x400000u : 0);
      cu_->write_vgpr(vb_ + 3, lane, gradient ? 0xbeef3800u : uv);
    }
    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.is_load = true;
    reads->bytes = {};
    ASSERT_TRUE(
        amdgpu::prepare_image_transfer(*wf_, state, 0, 8, coords, fields, 8, mode).succeeded());
    const unsigned level = gradient ? 2 : 1;
    ASSERT_TRUE(state.image_status);
    EXPECT_EQ(state.image_status->levels[0], level);
    const auto mip =
        amdgpu::image_mip_layout(arch() == ROCJITSU_CODE_ARCH_RDNA4, 0, 4, 64, height, 3, level);
    ASSERT_TRUE(mip);
    if (gradient) {
      EXPECT_EQ(state.per_lane_addr[0], kBase + mip->offset + 9 * 4);
    } else {
      ASSERT_TRUE(state.graphics_sample);
      ASSERT_TRUE(state.graphics_sample->comparison);
      EXPECT_EQ(state.graphics_sample->comparison->function, 1u);
      EXPECT_EQ(state.graphics_sample->comparison->references[0], 3.f);
      ASSERT_EQ(state.graphics_sample->tap_count, 4u);
      for (unsigned tap = 0; tap < 4; ++tap)
        EXPECT_EQ(state.graphics_sample->taps[tap].addresses[0],
                  kBase + mip->offset + 4 * (8 * mip->pitch + 7 + tap));
    }

    SampleInputs operation(*sample);
    auto &inputs = operation.inputs;
    inputs.image_coordinates = coords;
    inputs.image_a16 = true;
    inputs.image_coordinate_count = gradient ? 1 : 2;
    inputs.image_prefix_words = 3;
    inputs.image_prefix_low16_words =
        amdgpu::image_prefix_low16_words(gradient ? 1 : 3, gradient ? 1 : 0, true, !gradient, true);
    auto &controls = *inputs.image_sample_sources;
    controls.sampler = 8;
    controls.dim = dim;
    controls.mode = mode;
    controls.linear_candidate = false;
    controls.unorm = false;
    controls.offset = true;
    controls.compare = controls.gather = !gradient;
    const std::array<uint8_t, 5> expected{15, 3, uint8_t(gradient ? 3 : 15),
                                          uint8_t(gradient ? 3 : 15), 0};
    EXPECT_EQ(reads->bytes[0], expected);
    auto &waits = wf_->ensure_memory_wait_scoreboard();
    for (unsigned lane = 0; lane < 4; ++lane) {
      for (uint16_t reg = 0; reg < expected.size(); ++reg) {
        if (lane) {
          EXPECT_EQ(reads->bytes[lane][reg], !gradient && lane < 3 && reg == 3 ? 15 : 0);
        }
        for (uint8_t half : {0x3, 0xc}) {
          unsigned reports = 0;
          waits.clear();
          const auto counter = amdgpu::WaitCounterKind::Load;
          waits.add({waits.issue(counter),
                     0x100,
                     uint64_t{1} << lane,
                     {RegClass::VGPR, reg, 1},
                     counter,
                     half});
          waits.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
          waits.check_instruction(operation, *wf_);
          EXPECT_EQ(reports, unsigned((reads->bytes[lane][reg] & half) != 0))
              << gradient << "/" << lane << "/" << reg << "/" << +half;
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, Gfx10PackedPrefixesSelectTexelsAndConsumeTheirEncodedBytes) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA1 && arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP();
  class CoordinateReads final : public ExecutionPlugin {
  public:
    CoordinateReads() : ExecutionPlugin("image_coordinate_reads") {}
    std::array<uint8_t, 6> bytes{};
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t mask) override {
      const uint32_t index = reg - wave->vgpr_alloc().base;
      if ((lanes & 1) && index < bytes.size())
        bytes[index] |= mask;
    }
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<CoordinateReads>();
  auto *reads = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  wf_->set_exec(1);
  const auto coords = amdgpu::contiguous_image_address_registers(0);

  // A one-dimensional G16 derivative consumes only the low half of each gradient.
  bind_sample_image(
      Image{.width = 64, .pitch = 64, .type = amdgpu::kImage1d, .mip_levels = 3, .last_level = 2});
  bind_sampler({.mip = MipFilter::Point, .max_lod_fixed8 = 512});
  cu_->write_vgpr(vb_, 0, 0x1234); // Signed six-bit X offset, independent of the LOD calculation.
  cu_->write_vgpr(vb_ + 1, 0, 0xabcd2c00); // 1/16 of 64 texels selects mip 2.
  cu_->write_vgpr(vb_ + 2, 0, 0xdead0000);
  cu_->write_vgpr(vb_ + 3, 0, 0xbeef3800); // A16 X = 0.5.
  amdgpu::VectorMemState gradients(amdgpu::GLOBAL_MEM);
  gradients.is_load = true;
  gradients.dst_reg_base = vb_ + 8;
  gradients.texel_status = std::make_unique<amdgpu::TexelStatus>();
  ASSERT_TRUE(amdgpu::prepare_image_sample(
                  *wf_, gradients, 0, 8, coords,
                  {.dmask = 1, .dim = 0, .a16 = true, .gradient16 = true, .offset = true},
                  amdgpu::ImageSampleMode::Derivatives16)
                  .succeeded());
  EXPECT_EQ(gradients.texel_status->lod[0], 2u);
  EXPECT_EQ(reads->bytes, (std::array<uint8_t, 6>{15, 3, 3, 3, 0, 0}));

  // All four horizontal taps are adjacent X texels. The comparison reference
  // remains FP32 even when the bias and coordinate pair use FP16.
  bind_sample_image(Image{.width = 8, .height = 8});
  bind_sampler({.normalized = false});
  cu_->write_sgpr(sb_ + 8, cu_->read_sgpr(sb_ + 8) | (1u << 12)); // LESS
  cu_->write_vgpr(vb_, 0, 1);
  cu_->write_vgpr(vb_ + 1, 0, 0xdeed3c00); // Bias +1
  cu_->write_vgpr(vb_ + 2, 0, std::bit_cast<uint32_t>(3.0f));
  cu_->write_vgpr(vb_ + 3, 0, 0x3c004000); // X = 2, Y = 1
  reads->bytes.fill(0);
  amdgpu::VectorMemState gather(amdgpu::GLOBAL_MEM);
  gather.is_load = true;
  gather.dst_reg_base = vb_ + 8;
  gather.texel_status = std::make_unique<amdgpu::TexelStatus>();
  ASSERT_TRUE(amdgpu::prepare_image_sample(*wf_, gather, 0, 8, coords,
                                           {.dmask = 1,
                                            .dim = 1,
                                            .a16 = true,
                                            .gather = true,
                                            .offset = true,
                                            .lod_bias = true,
                                            .compare = true,
                                            .horizontal = true},
                                           amdgpu::ImageSampleMode::Bias)
                  .succeeded());
  ASSERT_EQ(gather.image_sample->taps.size(), 4u);
  for (unsigned tap = 0; tap < 4; ++tap) {
    auto &sample = gather.image_sample->taps[tap];
    EXPECT_EQ(sample.addresses[0], kBase + 4 * (2 + tap));
    sample.response.resize(wf_->wf_size() * sizeof(float));
    const float depth = float(2 + tap);
    std::memcpy(sample.response.data(), &depth, sizeof(depth));
  }
  amdgpu::complete_buffer_format_load(*wf_, *cu_, gather);
  for (unsigned tap = 0; tap < 4; ++tap)
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + tap, 0), tap >= 2 ? 0x3f800000u : 0u);
  EXPECT_EQ(reads->bytes, (std::array<uint8_t, 6>{15, 3, 15, 15, 0, 0}));

  bind_sample_image(Image{.width = 64, .height = 64, .mip_levels = 3, .last_level = 2});
  bind_sampler({.mip = MipFilter::Point, .max_lod_fixed8 = 512});
  wf_->set_exec(15);
  for (unsigned lane = 0; lane < 4; ++lane) {
    cu_->write_vgpr(vb_ + 1, lane, std::bit_cast<uint32_t>((8.f + (lane & 1)) / 64));
    cu_->write_vgpr(vb_ + 2, lane, std::bit_cast<uint32_t>((8.f + (lane >> 1)) / 64));
  }
  for (const float bias : {0.f, 1.f}) {
    for (unsigned lane = 0; lane < 4; ++lane)
      cu_->write_vgpr(vb_, lane, std::bit_cast<uint32_t>(bias));
    amdgpu::VectorMemState biased(amdgpu::GLOBAL_MEM);
    biased.is_load = true;
    biased.dst_reg_base = vb_ + 8;
    biased.texel_status = std::make_unique<amdgpu::TexelStatus>();
    ASSERT_TRUE(amdgpu::prepare_image_sample(*wf_, biased, 0, 8, coords,
                                             {.dmask = 1, .dim = 1, .lod_bias = true},
                                             amdgpu::ImageSampleMode::Bias)
                    .succeeded());
    EXPECT_EQ(biased.texel_status->lod[0], uint8_t(bias));
  }
}

TEST_P(ImageSamplerControlTest, Gfx9AndGfx10RejectUnqualifiedShaderLodClamp) {
  if (!arch_is_cdna_4_or_lower(arch()) && arch() != ROCJITSU_CODE_ARCH_RDNA1 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP();
  bind_sample_image(
      Image{.width = 64, .height = 64, .pitch = 64, .mip_levels = 2, .last_level = 1});
  bind_sampler({.mip = MipFilter::Point, .max_lod_fixed8 = 256, .bias_fixed8 = 256});
  wf_->set_exec(1);
  write_coordinates(0, {std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(0.5f), 0u});
  amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
  state.is_load = true;
  state.dst_reg_base = vb_ + 8;
  ASSERT_TRUE(amdgpu::prepare_image_sample(*wf_, state, 0, 8,
                                           amdgpu::contiguous_image_address_registers(0),
                                           {.dmask = 1, .dim = 1, .da = true, .lod_clamp = true},
                                           amdgpu::ImageSampleMode::Zero)
                  .failed());
  EXPECT_EQ(wf_->instruction_execution_error(),
            amdgpu::InstructionExecutionError::UnimplementedInstruction);
}

TEST_P(ImageSamplerControlTest, ShaderClampSelectsTheSameMipInBothPreparations) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  bind_sample_image(Image{.width = 64, .height = 64, .pitch = 1, .mip_levels = 2, .last_level = 1});
  bind_sampler({.mip = MipFilter::Point, .max_lod_fixed8 = 256});
  wf_->set_exec(1);
  constexpr amdgpu::ImageFields fields{.dmask = 1, .dim = 1, .da = true, .lod_clamp = true};
  const auto coords = amdgpu::contiguous_image_address_registers(0);
  for (float minimum : {0.f, 1.f}) {
    write_coordinates(0, {std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(0.5f),
                          std::bit_cast<uint32_t>(minimum)});
    amdgpu::VectorMemState compute(amdgpu::GLOBAL_MEM), transfer(amdgpu::GLOBAL_MEM);
    compute.is_load = transfer.is_load = true;
    compute.dst_reg_base = vb_ + 8;
    compute.texel_status = std::make_unique<amdgpu::TexelStatus>();
    ASSERT_TRUE(amdgpu::prepare_image_sample(*wf_, compute, 0, 8, coords, fields,
                                             amdgpu::ImageSampleMode::Zero)
                    .succeeded());
    ASSERT_EQ(compute.texel_status->lod[0], uint8_t(minimum));
    amdgpu::ImageLodResults query{};
    ASSERT_TRUE(amdgpu::prepare_image_transfer(*wf_, transfer, 0, 8, coords, fields, 8,
                                               amdgpu::ImageSampleMode::Zero, &query)
                    .succeeded());
    EXPECT_EQ(query[0][0], minimum);
    EXPECT_EQ(query[0][1], minimum);
  }
}

TEST_P(ImageSamplerControlTest, Gfx10WarningReportsIntegerLodBeforePointMipRounding) {
  const bool rdna1 = arch() == ROCJITSU_CODE_ARCH_RDNA1;
  if (!rdna1 && arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP();
  bind_image(
      {.width = 64, .height = 64, .pitch = 1, .mip_levels = 2, .last_level = 1, .rdna_format = 22});
  cu_->write_sgpr(sb_ + 5, cu_->read_sgpr(sb_ + 5) | (512u << 8));
  wf_->set_exec(1);
  write_coordinates(0, std::bit_cast<std::array<uint32_t, 3>>(std::array{0.f, 0.f, 0.75f}));
  for (unsigned mip = 0; mip <= 1; ++mip) {
    const auto layout = amdgpu::ImageAddressing::prepare(
        cu_->image_address_config(),
        {.width = 64, .height = 64, .element_bytes = 4, .mip_levels = 2}, mip);
    ASSERT_TRUE(layout);
    memory_.write32(kBase + layout->byte_offset(0, 0, 0), std::bit_cast<uint32_t>(float(mip)));
  }
  for (auto filter : {MipFilter::Point, MipFilter::None}) {
    bind_sampler({.mip = filter});
    cu_->write_vgpr(vb_ + 9, 0, 0xdeadbeef);
    auto decoder = Decoder::create(arch());
    auto inst =
        rdna1 ? decoder->decode_window(
                    rdna1::build_mimg(rdna1::kImageSampleLMimg,
                                      {.dim = 1, .dmask = 1, .lwe = 1, .vdata = 8, .ssamp = 2}))
              : decoder->decode_window(
                    rdna2::build_mimg(rdna2::kImageSampleLMimg,
                                      {.dim = 1, .dmask = 1, .lwe = 1, .vdata = 8, .ssamp = 2}));
    ASSERT_TRUE(inst.succeeded());
    ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), filter == MipFilter::Point ? 0x3f800000u : 0u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 2u);
  }
}

TEST_P(ImageSamplerControlTest, Gfx11Swizzle30UsesTheQualifiedAddressForOrdinaryAndFeedback) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  const bool qualified = cu_->image_address_config().gb_addr_config == 0x545;
  wf_->set_exec(1);
  bind_sampler();
  for (bool array : {false, true}) {
    const auto dim = array ? amdgpu::kImageDim2dArray : amdgpu::kImageDim2d;
    dimension_override_ = dim;
    for (auto base : {0x10000u, 0x10100u, 0x40000u}) {
      for (unsigned level : {0u, 1u}) {
        bind_image({.width = 64,
                    .height = 64,
                    .depth = array ? 2u : 1u,
                    .pitch = 1,
                    .type = array ? amdgpu::kImage2dArray : amdgpu::kImage2d,
                    .swizzle_mode = 30,
                    .mip_levels = 2,
                    .base_level = level,
                    .last_level = level});
        cu_->write_sgpr(sb_, base >> 8);
        const unsigned layer = array ? 1 : 0;
        // W7900 point samples returned distinct markers placed at these byte offsets.
        constexpr uint32_t offsets[2][2][2] = {{{0x2011c, 0x2090c}, {0x6111c, 0x6190c}},
                                               {{0x1021c, 0x10a0c}, {0x5121c, 0x51a0c}}};
        for (auto x : {3u, 17u}) {
          const uint32_t y = 1;
          const auto address = base + offsets[level][layer][x == 17];
          for (bool tfe : {false, true}) {
            SCOPED_TRACE(testing::Message() << "base=" << base << " level=" << level << " x=" << x
                                            << " layer=" << layer << " tfe=" << tfe);
            texel_fault_ = tfe;
            const bool supported = qualified && base == 0x40000;
            write_coordinates(0, {x, y, layer});
            const uint32_t value = 1 + 64 * level + 32 * layer + x;
            memory_.write32(address, 0xdeadbeef);
            for (bool load : {false, true}) {
              wf_->clear_instruction_execution_error();
              cu_->write_vgpr(vb_ + 8, 0, load ? 0xdeadbeef : value);
              ASSERT_EQ(decoded_issue(load, array ? 3 : 2), supported);
              if (supported) {
                EXPECT_EQ(memory_.read32(address), value);
                EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), value);
              }
            }
            const float size = float(64 >> level);
            write_coordinates(0, std::bit_cast<std::array<uint32_t, 3>>(std::array{
                                     (x + 0.5f) / size, (y + 0.5f) / size, float(layer)}));
            cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
            auto sample = modern_sample_lz(false, 1, tfe, dim);
            ASSERT_TRUE(sample.succeeded());
            wf_->clear_instruction_execution_error();
            ASSERT_EQ(cu_->execute_instruction(sample.value().get(), *wf_).succeeded(), supported);
            if (supported) {
              amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
              ASSERT_EQ(pipeline.issue(sample.value().release(), *wf_),
                        amdgpu::VmAccessOutcome::Complete);
              EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), value);
            }
          }
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, Gfx11CompressionWithoutMetadataStopsBeforeIssue) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  wf_->set_exec(1);
  for (bool volume : {false, true}) {
    bind_image({.width = 64,
                .height = 64,
                .depth = volume ? 8u : 1u,
                .pitch = 1,
                .type = volume ? amdgpu::kImage3d : amdgpu::kImage2d,
                .swizzle_mode = 27});
    cu_->write_sgpr(sb_ + 6, 1u << 21);
    write_coordinates(0, {8, 0, volume ? 1u : 0u});
    for (bool tfe : {false, true}) {
      texel_fault_ = tfe;
      for (bool load : {false, true}) {
        wf_->clear_instruction_execution_error();
        EXPECT_FALSE(decoded_issue(load, volume ? 3 : 2));
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, ResidentHalfArrayUsesNativeFilterPrecision) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  wf_->set_exec(1);
  bind_sampler({.linear_xy = true});
  write_coordinates(0, std::bit_cast<std::array<uint32_t, 3>>(std::array{0.375f, 0.375f, 0.f}));
  for (uint32_t swizzle : {0u, gfx12 ? 3u : 27u}) {
    bind_image({.width = 2,
                .height = 2,
                .depth = 2,
                .type = amdgpu::kImage2dArray,
                .swizzle_mode = swizzle,
                .rdna_format = 13});
    const auto layout = amdgpu::ImageAddressing::prepare(
        cu_->image_address_config(),
        {.width = 2, .height = 2, .depth = 2, .element_bytes = 2, .swizzle_mode = swizzle}, 0);
    ASSERT_TRUE(layout);
    const uint16_t texels[]{0, 0x3c00, 0x3bff, 0x3401};
    for (uint32_t i = 0; i < 4; ++i)
      memory_.write16(kBase + layout->byte_offset(i % 2, i / 2, 0), texels[i]);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    for (bool tfe : {false, true}) {
      SCOPED_TRACE(testing::Message() << "swizzle=" << swizzle << " tfe=" << tfe);
      cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
      cu_->write_vgpr(vb_ + 9, 0, 0xdeadbeef);
      ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 1, tfe, amdgpu::kImageDim2dArray));
      // Normalized bilinear result on a W7900 and R9700 for both TFE settings.
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x3ec7f400u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0xdeadbeefu);
      auto missing_layer = modern_sample_lz(false, 1, tfe, amdgpu::kImageDim2d);
      ASSERT_TRUE(missing_layer.succeeded());
      wf_->clear_instruction_execution_error();
      EXPECT_TRUE(cu_->execute_instruction(missing_layer.value().get(), *wf_).failed());
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_FALSE(missing_layer.value()->data());
      wf_->clear_instruction_execution_error();
    }
  }
}

TEST_P(ImageSamplerControlTest, OneDimensionalSamplesValidateTheFullDescriptorPitch) {
  if (!amdgpu::has_gfx11_plus_image_operands(arch()))
    GTEST_SKIP();
  wf_->set_exec(1);
  bind_sampler(); // Point filtering uses only X.
  write_coordinates(0, {std::bit_cast<uint32_t>(5.5f / 30), 0, 0});
  auto &waits = wf_->ensure_memory_wait_scoreboard();
  unsigned reports = 0;
  waits.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
  for (uint32_t format : {kR32Uint, 22u, 7u}) {
    const bool unorm16 = format == 7;
    const uint32_t value = unorm16 ? std::bit_cast<uint32_t>(1.f) : 0x41234567u;
    if (unorm16)
      memory_.write16(kBase + 10, 0xffff);
    else
      memory_.write32(kBase + 20, value);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    for (uint32_t pitch : {1u, 8u, 30u, 64u, 128u}) {
      bind_image({.width = 30, .pitch = pitch, .type = amdgpu::kImage1d, .rdna_format = format});
      for (bool r128 : {false, true}) {
        const bool valid = r128 || (pitch >= 30 && (arch() != ROCJITSU_CODE_ARCH_RDNA3_5 ||
                                                    pitch * (unorm16 ? 2 : 4) % 256 == 0));
        for (bool warn : {false, true}) {
          cu_->write_sgpr(sb_ + 5, uint32_t(warn) << (arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 5 : 8));
          for (bool tfe : {false, true}) {
            SCOPED_TRACE(testing::Message() << "format=" << format << " pitch=" << pitch << " r128="
                                            << r128 << " warn=" << warn << " tfe=" << tfe);
            auto decoded = modern_sample_lz(false, 1, tfe, amdgpu::kImageDim1d, r128);
            ASSERT_TRUE(decoded.succeeded());
            EXPECT_EQ(decoded.value()->src_operand(1)->size_bits(), r128 ? 128 : 256);
            EXPECT_NE(decoded.value()->disassemble().find("s[0:7]"), std::string::npos);
            reports = 0;
            waits.clear();
            constexpr auto counter = amdgpu::WaitCounterKind::Load;
            waits.add({waits.issue(counter), 0x100, 1, {RegClass::VGPR, 0, 1}, counter, 0x8});
            waits.check_instruction(*decoded.value(), *wf_);
            EXPECT_EQ(reports != 0, valid);
            waits.clear();
            reports = 0;
            waits.add({waits.issue(counter), 0x100, 1, {RegClass::SGPR, 4, 1}, counter, 0xf});
            waits.check_instruction(*decoded.value(), *wf_);
            EXPECT_EQ(reports != 0, !r128);
            waits.clear();
            constexpr uint32_t sentinel = 0xdeadbeef;
            fill_destination(sentinel);
            wf_->clear_instruction_execution_error();
            const auto result = cu_->execute_instruction(decoded.value().get(), *wf_);
            if (valid) {
              ASSERT_TRUE(result.succeeded());
              amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
              ASSERT_EQ(pipeline.issue(decoded.value().release(), *wf_),
                        amdgpu::VmAccessOutcome::Complete);
            } else {
              EXPECT_TRUE(result.failed());
              EXPECT_EQ(wf_->instruction_execution_error(),
                        amdgpu::InstructionExecutionError::UnsupportedOperandValue);
              EXPECT_FALSE(decoded.value()->data());
            }
            EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), valid ? value : sentinel);
            EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), sentinel);
          }
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, ResidentFloatAndUnormArraysPreserveValuesWithFeedback) {
  if (!amdgpu::has_gfx11_plus_image_operands(arch()))
    GTEST_SKIP();
  wf_->set_exec(1);
  bind_sampler({.linear_xy = true});
  auto &waits = wf_->ensure_memory_wait_scoreboard();
  unsigned reports = 0;
  waits.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
  constexpr auto counter = amdgpu::WaitCounterKind::Load;
  for (uint32_t format : {22u, 7u}) {
    const bool fp32 = format == 22;
    if (fp32) {
      memory_.write32(kBase, 0x1f082e19);
      memory_.write32(kBase + 4, 0x204a0a24);
    } else {
      memory_.write16(kBase, 0);
      memory_.write16(kBase + 2, 0xffff);
    }
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    const float u = fp32 ? std::bit_cast<float>(0x3f1b9f54u) * .25f : .1875f;
    write_coordinates(0, {std::bit_cast<uint32_t>(u), 0, 0});
    for (uint32_t dim : {amdgpu::kImageDim1d, amdgpu::kImageDim1dArray, amdgpu::kImageDim2dArray}) {
      for (uint32_t swizzle : {0u, arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 3u : 27u}) {
        if (swizzle && dim != amdgpu::kImageDim2dArray && arch() != ROCJITSU_CODE_ARCH_RDNA4)
          continue;
        bind_image({.width = 4,
                    .height = dim == amdgpu::kImageDim2dArray ? 4u : 1u,
                    .depth = dim == amdgpu::kImageDim1d ? 1u : 2u,
                    .pitch = 128,
                    .type = amdgpu::kImage1d + dim,
                    .swizzle_mode = swizzle,
                    .rdna_format = format});
        for (bool tfe : {false, true}) {
          SCOPED_TRACE(testing::Message() << "dim=" << dim << " format=" << format
                                          << " swizzle=" << swizzle << " tfe=" << tfe);
          auto decoded = modern_sample_lz(false, 1, tfe, dim);
          ASSERT_TRUE(decoded.succeeded());
          const uint32_t coordinates = amdgpu::image_coordinate_count(dim);
          for (uint16_t reg = 0; reg <= coordinates; ++reg) {
            for (uint64_t lane : {1u, 2u}) {
              reports = 0;
              waits.clear();
              waits.add(
                  {waits.issue(counter), 0x100, lane, {RegClass::VGPR, reg, 1}, counter, 0x8});
              waits.check_instruction(*decoded.value(), *wf_);
              EXPECT_EQ(reports != 0, reg < coordinates && lane == 1);
            }
          }
          waits.clear();
          constexpr uint32_t sentinel = 0xdeadbeef;
          cu_->write_vgpr(vb_ + 8, 0, sentinel);
          cu_->write_vgpr(vb_ + 9, 0, sentinel);
          ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 1, tfe, dim));
          // Captured on the W7900 and R9700 for both TFE settings.
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), fp32 ? 0x1f51ad7du : 0x3e800000u);
          EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), sentinel);
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, SamplerControlAdmissionAndSourceWaitsAgreeAcrossFeedback) {
  if (!amdgpu::has_gfx11_plus_image_operands(arch()))
    GTEST_SKIP();
  struct Control {
    const char *name;
    unsigned word, bits, perf_mip;
    bool supported;
  };
  const Control controls[] = {{"LOD_BIAS_SEC", 2, 1u << 14, 0, true},
                              {"SKIP_DEGAMMA", 0, 1u << 31, 0, true},
                              {"PERF_MOD + PERF_MIP", 0, 0, 8, true},
                              {"ANISO_OVERRIDE", 2, 1u << 29, 0, false},
                              {"DERIVATIVE_ADJUST", 2, 1u << 31, 0, false}};
  constexpr uint32_t texel = std::bit_cast<uint32_t>(0.75f), sentinel = 0xdeadbeef;
  memory_.write32(kBase, texel);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (const auto &c : controls) {
    if (c.word == 2 && c.bits == (1u << 31) && arch() == ROCJITSU_CODE_ARCH_RDNA4)
      continue; // GFX12 repurposes DERIVATIVE_ADJUST for PERF_MIP.
    bind_sample_image({.width = 4, .height = 4, .pitch = 64, .perf_mod = c.perf_mip ? 1u : 0u});
    bind_sampler({.perf_mip = c.perf_mip});
    cu_->write_sgpr(sb_ + 8 + c.word, cu_->read_sgpr(sb_ + 8 + c.word) | c.bits);
    for (bool tfe : {false, true}) {
      SCOPED_TRACE(testing::Message() << c.name << " tfe=" << tfe);
      auto inst = modern_sample_lz(false, 1, tfe);
      ASSERT_TRUE(inst.succeeded());
      auto &waits = wf_->ensure_memory_wait_scoreboard();
      waits.clear();
      constexpr auto counter = amdgpu::WaitCounterKind::Load;
      waits.add({waits.issue(counter), 0x100, 1, {RegClass::VGPR, 0, 1}, counter, 0xf});
      unsigned reports = 0;
      waits.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
      waits.check_instruction(*inst.value(), *wf_);
      EXPECT_EQ(reports != 0, c.supported);
      waits.clear();
      wf_->clear_instruction_execution_error();
      cu_->write_vgpr(vb_ + 8, 0, sentinel);
      cu_->write_vgpr(vb_ + 9, 0, sentinel);
      const auto outcome = cu_->execute_instruction(inst.value().get(), *wf_);
      if (c.supported) {
        ASSERT_TRUE(outcome.succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(inst.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
      } else {
        EXPECT_TRUE(outcome.failed());
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
        EXPECT_FALSE(inst.value()->data());
      }
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), c.supported ? texel : sentinel);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), sentinel);
    }
  }

  // On W7900 and R9700, SKIP_DEGAMMA returns the stored R8_SRGB byte as UNORM.
  bind_image({.width = 4, .height = 4, .pitch = 256, .rdna_format = 64});
  bind_sampler();
  cu_->write_sgpr(sb_ + 8, cu_->read_sgpr(sb_ + 8) | (1u << 31));
  memory_.write32(kBase, 0x80);
  cu_->l1_vector().invalidate_all();
  l2_.invalidate_all();
  for (bool tfe : {false, true}) {
    cu_->write_vgpr(vb_ + 8, 0, sentinel);
    ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 1, tfe));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x3f008081u);
  }
}

TEST_P(ImageSamplerControlTest, NativeBorderAndAnisotropyHaveConsistentAdmission) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA4 && arch() != ROCJITSU_CODE_ARCH_RDNA3 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  bind_image({.width = 4, .height = 4, .pitch = 64});
  wf_->set_exec(1);
  memory_.write32(kBase, 42);
  for (unsigned control : {0u, 1u, 2u, 3u}) {
    for (unsigned normalization : {0u, 1u, 2u}) {
      const bool unrm = normalization == 1;
      const float center = normalization ? .5f : .125f;
      write_coordinates(0, std::bit_cast<std::array<uint32_t, 3>>(std::array{center, center, 0.f}));
      bind_sampler({.clamp_x = Clamp::Border,
                    .clamp_y = Clamp::Border,
                    .max_lod_fixed8 = 0,
                    .normalized = normalization != 2});
      if (control == 1 || control == 2)
        cu_->write_sgpr(sb_ + 11, control << 30);
      if (control == 3)
        cu_->write_sgpr(sb_ + 8, cu_->read_sgpr(sb_ + 8) | (1u << 9));
      for (bool tfe : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << control << " normalization=" << normalization << " tfe=" << tfe);
        cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
        ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(unrm, 1, tfe));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 42u);
      }
    }
    if (arch() == ROCJITSU_CODE_ARCH_RDNA4) {
      // A resource with allocated mips but a single visible level distinguishes the
      // graphics sampler's implicit quad reads from the ordinary compute sampler's.
      bind_image({.width = 4, .height = 4, .pitch = 1, .mip_levels = 2});
      auto &waits = wf_->ensure_memory_wait_scoreboard();
      for (bool tfe : {false, true}) {
        rdna4::VsampleBuilderFields fields{};
        fields.dim = fields.dmask = 1;
        fields.vdata = fields.samp = 8;
        fields.vaddr1 = 1;
        fields.tfe = tfe;
        auto inst = Decoder::create(arch())->decode_window(
            rdna4::build_vsample(rdna4::kImageSampleVsample, fields));
        ASSERT_TRUE(inst.succeeded());
        waits.clear();
        constexpr auto counter = amdgpu::WaitCounterKind::Load;
        waits.add({waits.issue(counter), 0x100, 2, {RegClass::VGPR, 0, 1}, counter, 0xf});
        unsigned reports = 0;
        waits.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
        waits.check_instruction(*inst.value(), *wf_);
        EXPECT_EQ(reports != 0, tfe || control != 0);
        waits.clear();
      }
      bind_image({.width = 4, .height = 4, .pitch = 64});
    }
  }
}

TEST_P(ImageSamplerControlTest, SeamlessCubeIgnoresNonresidentZeroWeightTexels) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA4 && arch() != ROCJITSU_CODE_ARCH_RDNA3 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  class SparseCube final : public amdgpu::AddressSpaceTranslator {
  public:
    amdgpu::VmTranslationResult translate(uint64_t addr, size_t size,
                                          amdgpu::VmAccessKind access) const override {
      return amdgpu::IdentityAddressSpaceTranslator{}.translate(addr, size, access);
    }
    std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
      return std::make_shared<const amdgpu::PrtRanges>(
          amdgpu::PrtRanges{{kBase + 4096, kBase + 8192}, {kBase + 16384, kBase + 6 * 16384}});
    }
  };
  amdgpu::GpuVm vm;
  auto space =
      vm.register_address_space(1, std::make_shared<SparseCube>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_));
  ASSERT_TRUE(space);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(space);
  wf_->set_exec(1);
  bind_image(
      {.width = 64, .height = 64, .depth = 6, .type = amdgpu::kImageCube, .rdna_format = 22});
  bind_sampler({.linear_xy = true});
  // Center of (0,15) borders a missing row. Center of (63,63) borders two missing faces
  // and a three-face corner whose three helper texels all have zero weight.
  for (auto xy : {std::array{0u, 15u}, std::array{63u, 63u}}) {
    memory_.write32(kBase + 4 * (64 * xy[1] + xy[0]), 0x3f800000);
    const auto coordinates = std::array{1.f + (xy[0] + .5f) / 64, 1.f + (xy[1] + .5f) / 64, 0.f};
    write_coordinates(0, std::bit_cast<std::array<uint32_t, 3>>(coordinates));
    cu_->write_vgpr(vb_ + 9, 0, 0xdeadbeef);
    ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 1, true, amdgpu::kImageDimCube));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x3f800000u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0xdeadbeefu);
  }
  // A quarter-texel step in X gives the nonresident neighboring face positive weight.
  write_coordinates(0, std::bit_cast<std::array<uint32_t, 3>>(
                           std::array{1.f + 63.75f / 64, 1.f + 63.5f / 64, 0.f}));
  cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
  cu_->write_vgpr(vb_ + 9, 0, 0xdeadbeef);
  ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 1, true, amdgpu::kImageDimCube));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 0x3f400000u : 0u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 1u);
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(ImageSamplerControlTest, LayeredResourcesRequireALayerCoordinate) {
  if (!amdgpu::has_gfx11_plus_image_operands(arch()))
    GTEST_SKIP();
  wf_->set_exec(1);
  struct Case {
    uint32_t type, dim, swizzle, height;
  };
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  for (const auto test : {Case{amdgpu::kImageCube, amdgpu::kImageDim2dArray, gfx12 ? 3u : 27u, 64},
                          Case{amdgpu::kImageCube, amdgpu::kImageDim2dArray, 30u, 64},
                          Case{amdgpu::kImageCube, amdgpu::kImageDim2dArray, 31u, 64},
                          Case{amdgpu::kImage2dArray, amdgpu::kImageDim2d, gfx12 ? 3u : 27u, 64},
                          Case{amdgpu::kImage1dArray, amdgpu::kImageDim1d, 0, 1}}) {
    if (gfx12 && test.swizzle >= 30)
      continue;
    dimension_override_ = test.dim;
    bind_image({.width = 64,
                .height = test.height,
                .depth = 6,
                .type = test.type,
                .swizzle_mode = test.swizzle});
    write_coordinates(0, {3, 0, 0});
    for (bool tfe : {false, true}) {
      texel_fault_ = tfe;
      for (bool load : {false, true}) {
        SCOPED_TRACE(testing::Message() << test.type << " dim=" << test.dim << " swizzle="
                                        << test.swizzle << " tfe=" << tfe << " load=" << load);
        wf_->clear_instruction_execution_error();
        EXPECT_FALSE(decoded_issue(load, 3));
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnimplementedInstruction);
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, FeedbackReadsTheSameNonuniformTiledTexelAsAnOrdinarySample) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  const uint32_t config = cu_->image_address_config().gb_addr_config;
  ASSERT_TRUE(config == 0 || config == 0x545);
  bind_image(Image{.width = 256, .height = 256, .pitch = 1, .swizzle_mode = 27});
  bind_sampler();
  wf_->set_exec(1);
  // AddrLib's 256x256 R32 tile at (8,0): 0x100 on the zero configuration
  // and 0x2000 with GB_ADDR_CONFIG=0x545. Distinct texels expose a wrong route.
  memory_.write32(kBase + 0x100, 123);
  memory_.write32(kBase + 0x2000, 456);
  write_coordinates(0,
                    {std::bit_cast<uint32_t>(8.5f / 256), std::bit_cast<uint32_t>(0.5f / 256), 0});
  for (bool tfe : {false, true}) {
    ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 1, tfe));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), config ? 456u : 123u);
  }
}

TEST_P(ImageSamplerControlTest, BvhWaitTargetsItsOwnOutstandingCounter) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  wf_->wait_counters().increment(amdgpu::WaitCounterType::BVHCNT);
  wf_->set_wait_counter("wait_bvhcnt", 0);
  EXPECT_FALSE(wf_->wait_satisfied());
  wf_->wait_counters().decrement(amdgpu::WaitCounterType::BVHCNT);
  wf_->wait_counters().increment(amdgpu::WaitCounterType::VMCNT);
  EXPECT_TRUE(wf_->wait_satisfied());
  wf_->wait_counters().decrement(amdgpu::WaitCounterType::VMCNT);
  wf_->set_wait_counter("wait_bvhcnt", amdgpu::WaitCounters::BVHCNT_MAX);
  wf_->wait_counters().increment(amdgpu::WaitCounterType::BVHCNT);
  wf_->set_wait_all();
  EXPECT_FALSE(wf_->wait_satisfied());
  wf_->wait_counters().decrement(amdgpu::WaitCounterType::BVHCNT);
  EXPECT_TRUE(wf_->wait_satisfied());
}

TEST_P(ImageSamplerControlTest, Fp32LinearFilteringUsesTheSamePrecisionForEquivalentCoordinates) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();

  struct Texture {
    uint32_t format;
    std::array<uint32_t, 8> texels;
    std::array<uint32_t, 4> result;
  };
  // Two adjacent texels from the top row and the four result words captured on W7900 and R9700.
  const Texture textures[] = {
      {22, {0x1f082e19, 0x204a0a24}, {0x1f51ad7d, 0x1f51ad7d, 0x1f51ad7d, 0x1f51ad7d}},
      {50,
       {0x3aa10c93, 0x197f1252, 0x001f0438, 0x04f32650},
       {0x3a8f6f33, 0x19632c51, 0x3a8f6f33, 0x19632c51}},
      {63,
       {0x07fc052e, 0x2c1109e4, 0x0fdc2d30, 0x0e56107c, 0x32930aac, 0x39b033a6, 0x2d7613a0,
        0x12b412e9},
       {0x3100a956, 0x381a2d31, 0x2bd7512c, 0x11208b25}},
  };
  wf_->set_exec(1);
  for (const auto &texture : textures) {
    const uint32_t explicit_pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 32u : 64u;
    for (const uint32_t pitch : {1u, explicit_pitch}) {
      bind_image(Image{.width = 4, .height = 4, .pitch = pitch, .rdna_format = texture.format});
      // Request XYZW even when the stored texel only has one or two channels.
      cu_->write_sgpr(sb_ + 3, (cu_->read_sgpr(sb_ + 3) & ~0xfffu) | 0xfacu);
      cu_->l1_vector().flush_all();
      l2_.flush_all();
      for (size_t i = 0; i < texture.texels.size(); ++i)
        memory_.write32(kBase + 4 * i, texture.texels[i]);
      for (bool unrm : {false, true}) {
        for (bool force : {false, true}) {
          SCOPED_TRACE(testing::Message()
                       << "format=" << texture.format << " unrm=" << unrm << " force=" << force);
          bind_sampler({.linear_xy = true, .normalized = !force});
          const float scale = unrm || force ? 1.0f : 0.25f;
          write_coordinates(0, {std::bit_cast<uint32_t>(std::bit_cast<float>(0x3f1b9f54u) * scale),
                                std::bit_cast<uint32_t>(std::bit_cast<float>(0xbfb9fbe8u) * scale),
                                0});
          for (const bool tfe : {false, true}) {
            ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(unrm, 0xf, tfe));
            for (uint32_t c = 0; c < texture.result.size(); ++c)
              EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + c, 0), texture.result[c]);
          }
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, ResidentIntegerSamplesRepeatStoredChannelsWithAndWithoutFeedback) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  wf_->set_exec(1);
  bind_sampler();
  write_coordinates(0, {std::bit_cast<uint32_t>(0.125f), std::bit_cast<uint32_t>(0.125f), 0});
  for (const uint32_t format : {20u, 48u}) {
    bind_image(
        Image{.width = 4, .height = 4, .pitch = 64, .selectors = kIdentity, .rdna_format = format});
    cu_->write_sgpr(sb_ + 3, (cu_->read_sgpr(sb_ + 3) & ~0xfffu) | kIdentity);
    memory_.write32(kBase, 123);
    memory_.write32(kBase + 4, 456);
    // Physical W7900 and R9700 point samples select XYXY for RG32_UINT.
    const std::array<uint32_t, 4> expected{123, format == 20 ? 123u : 456u, 123,
                                           format == 20 ? 123u : 456u};
    for (const bool tfe : {false, true}) {
      ASSERT_NO_FATAL_FAILURE(issue_modern_sample_lz(false, 15, tfe));
      for (uint32_t index = 0; index < expected.size(); ++index)
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + index, 0), expected[index]);
    }
  }
}

TEST_P(ImageSamplerControlTest, Gfx12VolumeSampleFeedbackUsesActualHelperReads) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  class Reads final : public ExecutionPlugin {
  public:
    Reads() : ExecutionPlugin("volume_sampler_reads") {}
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t) override {
      if (reg == wave->vgpr_alloc().base)
        read_lanes |= lanes;
    }
    uint64_t read_lanes = 0;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Reads>();
  auto *reads = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  bind_image(Image{.width = 4,
                   .height = 4,
                   .depth = 4,
                   .pitch = 1,
                   .type = amdgpu::kImage3d,
                   .mip_levels = 2,
                   .last_level = 1,
                   .rdna_format = 22});
  for (unsigned lane = 0; lane < 4; ++lane)
    write_coordinates(lane, {lane == 1 ? 0x3f000000u : 0u, 0, 0});
  auto decoder = Decoder::create(arch());
  auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
  for (unsigned flags : {0u, 1u, 2u, 3u}) {
    for (bool mip_filter : {false, true}) {
      SCOPED_TRACE(testing::Message() << "feedback=" << flags << " mip-filter=" << mip_filter);
      wf_->set_exec(1);
      bind_sampler({.mip = mip_filter ? MipFilter::Point : MipFilter::None});
      rdna4::VsampleBuilderFields fields{};
      fields.dim = 2;
      fields.dmask = 1;
      fields.vdata = fields.samp = 8;
      fields.tfe = flags & 1;
      fields.lwe = flags >> 1;
      fields.vaddr1 = 1;
      fields.vaddr2 = 2;
      const auto encoded = rdna4::build_vsample(rdna4::kImageSampleVsample, fields);
      auto decoded = decoder->decode_window(encoded);
      ASSERT_TRUE(decoded.succeeded());
      unsigned warnings = 0;
      scoreboard.clear();
      const auto kind = amdgpu::WaitCounterKind::Ds;
      scoreboard.add({scoreboard.issue(kind), 0x100, 2, {RegClass::VGPR, 0, 1}, kind, 0xf});
      scoreboard.bind(0x200, &warnings,
                      [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
      scoreboard.check_instruction(*decoded.value(), *wf_);
      reads->read_lanes = 0;
      ASSERT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).succeeded());
      const uint64_t sample_reads = reads->read_lanes;
      const unsigned sample_warnings = warnings;
      wf_->set_exec(2);
      auto consume =
          decoder->decode_window(rdna4::build_vop1(rdna4::kVMovB32Vop1, {.src0 = 256, .vdst = 12}));
      ASSERT_TRUE(consume.succeeded());
      scoreboard.bind(0x204, &warnings,
                      [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
      scoreboard.check_instruction(*consume.value(), *wf_);
      reads->read_lanes = 0;
      ASSERT_TRUE(cu_->execute_instruction(consume.value().get(), *wf_).succeeded());
      EXPECT_EQ(reads->read_lanes, 2u);
      const unsigned consumer_warnings = warnings - sample_warnings;
      EXPECT_EQ(sample_reads, mip_filter ? 7u : 1u);
      EXPECT_EQ(sample_warnings, unsigned(bool(sample_reads & 2)));
      EXPECT_EQ(consumer_warnings, unsigned(!(sample_reads & 2)));
    }
  }
}

TEST_P(ImageSamplerControlTest, TrilinearSamplingIssuesOnlyContributingMipLevels) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  wf_->set_exec(1);
  auto decoder = Decoder::create(arch());
  rdna4::VsampleBuilderFields modern{};
  modern.dim = modern.dmask = modern.tfe = 1;
  modern.vdata = modern.samp = 8;
  modern.vaddr1 = 1;
  modern.vaddr2 = 2;
  rdna3_5::MimgBuilderFields legacy{};
  legacy.dim = legacy.dmask = legacy.tfe = 1;
  legacy.vdata = 8;
  legacy.ssamp = 2;
  struct Case {
    unsigned levels;
    float lod, coordinate;
    bool lower, upper;
    uint32_t perf_mip = 0;
  };
  for (auto c : {Case{4, 0, 0.3f, true, false}, Case{4, 0.125f, 0.3f, true, true},
                 Case{4, 1, 0.3f, true, false}, Case{1, 0.75f, 0.3f, true, false},
                 Case{4, 0.125f, -2.0f, false, false}, Case{4, 0.25f, 0.3f, true, false, 15},
                 Case{4, 0.5f, 0.3f, true, true, 15}, Case{4, 0.75f, 0.3f, false, true, 15}}) {
    SCOPED_TRACE(testing::Message() << c.levels << " mips, LOD=" << c.lod << " u=" << c.coordinate);
    bind_sample_image({.width = 64,
                       .height = 64,
                       .pitch = 1,
                       .mip_levels = c.levels,
                       .last_level = c.levels - 1,
                       .perf_mod = c.perf_mip ? 7u : 0u});
    bind_sampler({.clamp_x = Clamp::Border,
                  .clamp_y = Clamp::Border,
                  .mip = MipFilter::Linear,
                  .perf_mip = c.perf_mip});
    write_coordinates(0, {std::bit_cast<uint32_t>(c.coordinate),
                          std::bit_cast<uint32_t>(c.coordinate), std::bit_cast<uint32_t>(c.lod)});
    auto inst =
        gfx12 ? decoder->decode_window(rdna4::build_vsample(rdna4::kImageSampleLVsample, modern))
              : decoder->decode_window(rdna3_5::build_mimg(rdna3_5::kImageSampleLMimg, legacy));
    ASSERT_TRUE(inst.succeeded());
    ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
    auto *state = inst.value()->data_as<amdgpu::VectorMemState>();
    ASSERT_NE(state, nullptr);
    ASSERT_NE(state->graphics_sample, nullptr);
    const auto &sample = *state->graphics_sample;
    ASSERT_EQ(sample.tap_count, 8u);
    ASSERT_NE(state->image_status, nullptr);
    EXPECT_EQ(state->image_status->levels[0],
              c.levels == 1 ? 0u : uint32_t(c.lod) + unsigned(c.upper && !c.lower));
    if (c.upper && !c.lower) {
      EXPECT_EQ(state->per_lane_addr[0], sample.taps[4].addresses[0]);
    }
    for (unsigned i = 0; i < sample.tap_count; ++i)
      EXPECT_EQ(sample.taps[i].lane_mask, i < 4 ? uint64_t(c.lower) : uint64_t(c.upper));
  }
}

TEST_P(ImageSamplerControlTest, LinearSamplingSkipsZeroWeightNeighborsAcrossSparseTiles) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  bind_sample_image({.width = 256, .height = 256, .pitch = 1}, true);
  bind_sampler({.clamp_x = Clamp::Border, .clamp_y = Clamp::Border, .linear_xy = true});
  auto decoder = Decoder::create(arch());
  wf_->set_exec(1);
  rdna4::VsampleBuilderFields modern{};
  modern.dim = modern.dmask = modern.tfe = 1;
  modern.vdata = modern.samp = 8;
  modern.vaddr1 = 1;
  rdna3_5::MimgBuilderFields legacy{};
  legacy.dim = legacy.dmask = legacy.tfe = 1;
  legacy.vdata = 8;
  legacy.ssamp = 2;
  struct Case {
    float x, y;
    uint32_t taps;
  };
  for (auto c :
       {Case{127.5f, 127.5f, 1}, Case{127.5f + 1 / 512.f, 127.5f, 1},
        Case{127.5f + 1 / 256.f, 127.5f, 3}, Case{127.5f, 127.5f + 1 / 256.f, 5},
        Case{127.75f, 127.75f, 15}, Case{127.5f - 1 / 1024.f, 127.5f, 2}, Case{-1.5f, -1.5f, 0}}) {
    write_coordinates(0,
                      {std::bit_cast<uint32_t>(c.x / 256), std::bit_cast<uint32_t>(c.y / 256), 0});
    for (bool gather : {false, true}) {
      SCOPED_TRACE(testing::Message() << "x=" << c.x << " y=" << c.y << " gather=" << gather);
      auto inst =
          gfx12
              ? decoder->decode_window(rdna4::build_vsample(
                    gather ? rdna4::kImageGather4LzVsample : rdna4::kImageSampleLzVsample, modern))
              : decoder->decode_window(rdna3_5::build_mimg(
                    gather ? rdna3_5::kImageGather4LzMimg : rdna3_5::kImageSampleLzMimg, legacy));
      ASSERT_TRUE(inst.succeeded());
      ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
      const auto *state = inst.value()->data_as<amdgpu::VectorMemState>();
      ASSERT_NE(state, nullptr);
      ASSERT_NE(state->graphics_sample, nullptr);
      const auto &sample = *state->graphics_sample;
      ASSERT_GE(sample.tap_count, 4u);
      uint32_t active = 0;
      for (size_t i = 0; i < 4; ++i)
        active |= uint32_t(sample.taps[i].lane_mask & 1) << i;
      EXPECT_EQ(active, gather && c.taps ? 15u : c.taps);
    }
  }
}

TEST_P(ImageSamplerControlTest, SparseSampleAndGatherHonorTheGfx11BlendPrtBit) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && !amdgpu::has_gfx11_plus_image_operands(arch()))
    GTEST_SKIP();
  class SparseColumn final : public amdgpu::AddressSpaceTranslator {
  public:
    amdgpu::VmTranslationResult translate(uint64_t address, size_t bytes,
                                          amdgpu::VmAccessKind access) const override {
      return amdgpu::IdentityAddressSpaceTranslator{}.translate(address, bytes, access);
    }
    std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
      return std::make_shared<const amdgpu::PrtRanges>(
          amdgpu::PrtRanges{{kBase + 128 * 4, kBase + 129 * 4},
                            {kBase + (1024 + 128) * 4, kBase + (1024 + 129) * 4}});
    }
  };
  amdgpu::GpuVm vm;
  auto handle =
      vm.register_address_space(1, std::make_shared<SparseColumn>(),
                                std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_));
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  bind_sample_image(
      {.width = 1024, .height = 1024, .pitch = 1024, .mip_levels = 1, .last_level = 0}, true);
  wf_->set_exec(1);
  memory_.write32(kBase + 127 * 4, std::bit_cast<uint32_t>(4.f));
  memory_.write32(kBase + (1024 + 127) * 4, std::bit_cast<uint32_t>(8.f));
  write_coordinates(
      0, {std::bit_cast<uint32_t>(128.f / 1024), std::bit_cast<uint32_t>(0.5f / 1024), 0});
  auto decoder = Decoder::create(arch());
  for (bool blend : {false, true}) {
    if (gfx12 && !blend)
      continue;
    bind_sampler({.linear_xy = true});
    if (!gfx12 && blend)
      cu_->write_sgpr(sb_ + 10, cu_->read_sgpr(sb_ + 10) | (1u << 30));
    for (bool gather : {false, true})
      for (bool tfe : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "blend=" << blend << " tfe=" << tfe << " gather=" << gather);
        rdna3_5::MimgBuilderFields g11{};
        g11.dim = g11.dmask = 1;
        g11.tfe = tfe;
        g11.vdata = 8;
        g11.ssamp = 2;
        rdna4::VsampleBuilderFields g12{};
        g12.dim = g12.dmask = 1;
        g12.tfe = tfe;
        g12.vdata = g12.samp = 8;
        g12.vaddr1 = 1;
        auto inst =
            gfx12 ? decoder->decode_window(rdna4::build_vsample(
                        gather ? rdna4::kImageGather4LzVsample : rdna4::kImageSampleLzVsample, g12))
                  : decoder->decode_window(rdna3_5::build_mimg(
                        gather ? rdna3_5::kImageGather4LzMimg : rdna3_5::kImageSampleLzMimg, g11));
        ASSERT_TRUE(inst.succeeded());
        for (unsigned i = 0; i < 5; ++i)
          cu_->write_vgpr(vb_ + 8 + i, 0, 0xdeadbeefu);
        ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(inst.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
        const std::array<float, 4> expected =
            gather ? std::array{8.f, 0.f, 0.f, 4.f} : std::array{2.f, 0.f, 0.f, 0.f};
        for (unsigned i = 0; i < (gather ? 4u : 1u); ++i)
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + i, 0),
                    std::bit_cast<uint32_t>(blend ? expected[i] : 0.f));
        EXPECT_EQ(cu_->read_vgpr(vb_ + (gather ? 12 : 9), 0), tfe ? 1u : 0xdeadbeefu);
      }
  }
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(ImageSamplerControlTest, SparseFeedbackUsesPublishedPagesAndReportsTheRequestedMip) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  class Pages final : public amdgpu::AddressSpaceTranslator {
  public:
    bool resident = true, explicit_hole = false;
    std::shared_ptr<const amdgpu::PrtRanges> nonresident_prt_ranges() const override {
      static const auto hole =
          std::make_shared<const amdgpu::PrtRanges>(amdgpu::PrtRanges{{kBase, kBase + 4096}});
      return explicit_hole ? hole : nullptr;
    }
    amdgpu::IdentityAddressSpaceTranslator identity;
    amdgpu::VmTranslationResult translate(uint64_t address, size_t bytes,
                                          amdgpu::VmAccessKind kind) const override {
      return resident ? identity.translate(address, bytes, kind) : amdgpu::VmTranslationResult{};
    }
  };
  amdgpu::GpuVm vm;
  auto pages = std::make_shared<Pages>();
  unsigned faults = 0;
  auto handle = vm.register_address_space(
      1, pages, std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(memory_),
      [&](uint64_t, amdgpu::VmAccessKind) { ++faults; });
  ASSERT_TRUE(handle);
  cu_->set_gpu_vm(&vm);
  wf_->set_address_space(handle);
  wf_->set_exec(1);

  constexpr uint32_t sentinel = 0xdeadbeef;
  constexpr auto one = std::bit_cast<uint32_t>(1.0f);
  for (uint32_t address = 0; address < 4096; address += 4)
    memory_.write32(kBase + address, one);
  auto decoder = Decoder::create(arch());
  for (auto filter : {MipFilter::Point, MipFilter::Linear}) {
    bind_sampler({.mip = filter});
    for (bool volume : {false, true}) {
      bind_image(Image{.width = 4,
                       .height = 4,
                       .depth = volume ? 4u : 1u,
                       .pitch = 1,
                       .type = volume ? amdgpu::kImage3d : amdgpu::kImage2d,
                       .mip_levels = 2,
                       .last_level = 1,
                       .rdna_format = 22});
      const uint8_t dim = volume ? 2 : 1;
      rdna3_5::MimgBuilderFields gfx11{};
      gfx11.dim = dim;
      gfx11.dmask = 1;
      gfx11.vdata = 8;
      gfx11.ssamp = 2;
      rdna4::VsampleBuilderFields gfx12_fields{};
      gfx12_fields.dim = dim;
      gfx12_fields.dmask = 1;
      gfx12_fields.vdata = gfx12_fields.samp = 8;
      gfx12_fields.vaddr1 = 1;
      gfx12_fields.vaddr2 = 2;
      gfx12_fields.vaddr3 = 3;
      const auto decode = [&] {
        return gfx12
                   ? decoder->decode_window(
                         rdna4::build_vsample(rdna4::kImageSampleLVsample, gfx12_fields))
                   : decoder->decode_window(rdna3_5::build_mimg(rdna3_5::kImageSampleLMimg, gfx11));
      };
      for (unsigned flags : {0u, 1u, 2u, 3u}) {
        gfx11.tfe = gfx12_fields.tfe = flags & 1u;
        gfx11.lwe = gfx12_fields.lwe = flags >> 1;
        for (unsigned threshold : {0u, 1u, 256u}) {
          cu_->write_sgpr(sb_ + 5, threshold << (gfx12 ? 5 : 8));
          for (float lod : {0.25f, 0.75f, 1.0f}) {
            write_coordinates(0, {0, 0, 0});
            cu_->write_vgpr(vb_ + (volume ? 3 : 2), 0, std::bit_cast<uint32_t>(lod));
            for (unsigned mapping = 0; mapping < 4; ++mapping) {
              pages->resident = (mapping & 1u) == 0;
              pages->explicit_hole = mapping >= 2;
              const bool resident = pages->resident && !pages->explicit_hole;
              SCOPED_TRACE(testing::Message() << "volume=" << volume << " flags=" << flags
                                              << " threshold=" << threshold << " lod=" << lod
                                              << " resident=" << resident);
              faults = 0;
              fill_destination(sentinel);
              auto inst = decode();
              ASSERT_TRUE(inst.succeeded());
              ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
              amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
              const bool fault = !pages->resident && !pages->explicit_hole;
              EXPECT_EQ(pipeline.issue(inst.value().release(), *wf_),
                        fault ? amdgpu::VmAccessOutcome::Faulted
                              : amdgpu::VmAccessOutcome::Complete);
              EXPECT_EQ(faults, unsigned(fault));
              EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), fault ? sentinel : resident ? one : 0u);
              uint32_t status = sentinel;
              const uint32_t selected_level =
                  static_cast<uint32_t>(lod + (filter == MipFilter::Point ? 0.5f : 0.0f));
              if (pages->explicit_hole && (flags & 1))
                status = (selected_level << 16) | 1;
              else if (resident && (flags & 2) && lod * 256 < threshold)
                status = (selected_level << 16) | 2;
              EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), status);
            }
          }
        }
      }
    }
  }
  wf_->set_address_space({});
  cu_->set_gpu_vm(nullptr);
}

TEST_P(ImageSamplerControlTest, Gfx12SampleFeedbackUsesTheResourceWarningThreshold) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  constexpr uint32_t one = 0x3f800000, sentinel = 0xdeadbeef;
  bind_image(Image{
      .width = 4, .height = 4, .pitch = 1, .mip_levels = 2, .last_level = 1, .rdna_format = 22});
  bind_sampler({.mip = MipFilter::Point});
  for (uint32_t offset = 0; offset < 4096; offset += 4)
    memory_.write32(kBase + offset, one);
  for (unsigned lane = 0; lane < 4; ++lane)
    write_coordinates(lane, {one, one, 0});
  wf_->set_exec(1);
  auto read_flags = Decoder::create(arch())->decode_window(
      rdna4::build_sopk(rdna4::kSGetregB32Sopk, {.simm16 = 18 | (31 << 6), .sdst = 16}));
  ASSERT_TRUE(read_flags.succeeded());
  for (const auto op : {rdna4::kImageSampleLVsample}) {
    const unsigned words = op == rdna4::kImageSampleLVsample ? 1 : 4;
    for (unsigned flags : {0u, 1u, 2u, 3u}) {
      wf_->set_gfx12_excp_flag_user_extra_raw(1u << 30);
      bool sticky = false;
      for (bool warn : {false, true, false}) {
        SCOPED_TRACE(testing::Message() << op << " flags=" << flags << " warn=" << warn);
        // The low fractional bit of the GFX12 u5.8 MIN_LOD_WARN field is T# bit 165.
        cu_->write_sgpr(sb_ + 5, uint32_t(warn) << 5);
        for (unsigned i = 0; i <= words; ++i) {
          cu_->write_vgpr(vb_ + 8 + i, 0, sentinel);
          cu_->write_vgpr(vb_ + 8 + i, 1, sentinel);
        }
        rdna4::VsampleBuilderFields fields{};
        fields.dim = fields.dmask = 1;
        fields.vdata = fields.samp = 8;
        fields.tfe = flags & 1;
        fields.lwe = flags >> 1;
        fields.vaddr1 = 1;
        fields.vaddr2 = 2;
        auto instruction = Decoder::create(arch())->decode_window(rdna4::build_vsample(op, fields));
        ASSERT_TRUE(instruction.succeeded());
        ASSERT_TRUE(cu_->execute_instruction(instruction.value().get(), *wf_).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(instruction.value().release(), *wf_),
                  amdgpu::VmAccessOutcome::Complete);
        for (unsigned i = 0; i < words; ++i)
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + i, 0), one);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + words, 0), (warn && (flags & 2)) ? 2u : sentinel);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + words, 1), sentinel);
        sticky |= warn && (flags & 2);
        ASSERT_TRUE(cu_->execute_instruction(read_flags.value().get(), *wf_).succeeded());
        EXPECT_EQ(cu_->read_sgpr(sb_ + 16), uint32_t(sticky));
        EXPECT_NE(wf_->gfx12_excp_flag_user_extra_raw() & (1u << 30), 0u);
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, DroppedFilteredSamplesStillWriteLodWarnings) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  wf_->set_exec(1);
  write_coordinates(0, {std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(0.5f), 0});
  memory_.write32(kBase, 0xffff0000u);
  auto decoder = Decoder::create(arch());
  for (uint32_t format : {22u, 7u}) {
    bind_image(Image{.width = 4,
                     .height = 1,
                     .pitch = 1,
                     .mip_levels = 2,
                     .last_level = 1,
                     .rdna_format = format});
    cu_->write_sgpr(sb_ + 5, 1u << (gfx12 ? 5 : 8));
    for (bool linear : {false, true}) {
      bind_sampler({.linear_xy = linear, .mip = MipFilter::Point, .normalized = false});
      struct TestCase {
        uint8_t mask;
        bool lwe, tfe;
      };
      for (auto [mask, lwe, tfe] : {TestCase{1, true, false}, TestCase{0, false, false},
                                    TestCase{0, true, false}, TestCase{0, true, true}}) {
        SCOPED_TRACE(testing::Message()
                     << "arch=" << arch() << " format=" << format << " linear=" << linear
                     << " dmask=" << unsigned(mask) << " lwe=" << lwe << " tfe=" << tfe);
        rdna4::VsampleBuilderFields modern{};
        modern.dim = modern.unorm = 1;
        modern.dmask = mask;
        modern.lwe = lwe;
        modern.tfe = tfe;
        modern.vdata = modern.samp = 8;
        modern.vaddr1 = 1;
        rdna3_5::MimgBuilderFields legacy{};
        legacy.dim = legacy.unorm = 1;
        legacy.dmask = mask;
        legacy.lwe = lwe;
        legacy.tfe = tfe;
        legacy.vdata = 8;
        legacy.ssamp = 2;
        const auto modern_words = rdna4::build_vsample(rdna4::kImageSampleLzVsample, modern);
        const auto legacy_words = rdna3_5::build_mimg(rdna3_5::kImageSampleLzMimg, legacy);
        const std::span<const uint32_t> words = gfx12 ? std::span<const uint32_t>(modern_words)
                                                      : std::span<const uint32_t>(legacy_words);
        auto decoded = decoder->decode_window(words);
        ASSERT_TRUE(decoded.succeeded());
        ASSERT_EQ(decoded.value()->mnemonic(), "image_sample_lz");
        cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeefu);
        cu_->write_vgpr(vb_ + 9, 0, 0xdeadbeefu);
        wf_->clear_instruction_execution_error();
        const auto result = cu_->execute_instruction(decoded.value().get(), *wf_);
        ASSERT_TRUE(result.succeeded());
        auto *state = decoded.value()->data_as<amdgpu::VectorMemState>();
        ASSERT_NE(state, nullptr);
        if (!mask) {
          EXPECT_EQ(state->lane_mask, 0u);
        }
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(decoded.value().release(), *wf_),
                  amdgpu::VmAccessOutcome::Complete);
        const uint32_t data = cu_->read_vgpr(vb_ + 8, 0);
        const uint32_t status = cu_->read_vgpr(vb_ + 9, 0);
        if (!mask) {
          EXPECT_EQ(data, 0u);
        }
        EXPECT_EQ(status, lwe ? 2u : 0xdeadbeefu);
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, ResidentSampleStatusKeepsTheOlderWaitDependency) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  bind_image({.width = 2, .height = 1, .pitch = 1, .rdna_format = 7});
  bind_sampler({.linear_xy = true, .normalized = false});
  memory_.write32(kBase, 0xffff0000u);
  wf_->set_exec(1);
  write_coordinates(0, {std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(0.5f), 0});
  rdna3_5::MimgBuilderFields legacy{};
  legacy.dim = legacy.dmask = legacy.tfe = legacy.unorm = 1;
  legacy.vdata = 8;
  legacy.ssamp = 2;
  rdna4::VsampleBuilderFields modern{};
  modern.dim = modern.dmask = modern.tfe = modern.unorm = 1;
  modern.vdata = modern.samp = 8;
  modern.vaddr1 = 1;
  std::vector<uint32_t> fetch;
  if (gfx12) {
    const auto words = rdna4::build_vsample(rdna4::kImageSampleLzVsample, modern);
    fetch.assign(words.begin(), words.end());
  } else {
    const auto words = rdna3_5::build_mimg(rdna3_5::kImageSampleLzMimg, legacy);
    fetch.assign(words.begin(), words.end());
  }
  auto &wait = wf_->ensure_memory_wait_scoreboard();
  uint64_t pc = 0x8000;
  for (bool older : {false, true}) {
    for (bool payload : {false, true}) {
      SCOPED_TRACE(testing::Message() << "older=" << older << " payload=" << payload);
      pc += 0x100;
      wf_->pc = pc;
      wait.clear();
      if (older) {
        const auto counter = amdgpu::WaitCounterKind::Ds;
        wait.add({wait.issue(counter), pc - 4, 1, {RegClass::VGPR, 9, 1}, counter, 0xf});
      }
      cu_->write_vgpr(vb_ + 9, 0, 0xdeadbeefu);
      auto code = fetch;
      const auto src = uint16_t(256 + (payload ? 8 : 9));
      code.push_back(gfx12
                         ? rdna4::build_vop1(rdna4::kVMovB32Vop1, {.src0 = src, .vdst = 3})[0]
                         : rdna3_5::build_vop1(rdna3_5::kVMovB32Vop1, {.src0 = src, .vdst = 3})[0]);
      memory_.load_image(reinterpret_cast<const uint8_t *>(code.data()), code.size() * 4, pc);
      const auto before = cu_->memory_wait_diagnostic_count();
      ASSERT_TRUE(cu_->step());
      ASSERT_EQ(wf_->pc, pc + fetch.size() * 4);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(0.5f));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0xdeadbeefu);
      EXPECT_EQ(cu_->memory_wait_diagnostic_count(), before);
      ASSERT_TRUE(cu_->step());
      EXPECT_EQ(cu_->memory_wait_diagnostic_count() - before, unsigned(payload || older));
    }
  }
}

TEST_P(ImageSamplerControlTest, ResidentFilteredSamplesPreserveValuesWithFeedback) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  auto decoder = Decoder::create(arch());
  wf_->set_exec(1);
  const uint16_t values[]{0, 65535, 0x1234, 0x8765};
  memory_.write32(kBase, uint32_t(values[0]) | (uint32_t(values[1]) << 16));
  memory_.write32(kBase + 4, uint32_t(values[2]) | (uint32_t(values[3]) << 16));
  bind_sampler({.linear_xy = true, .normalized = false});
  for (uint32_t pitch : {1u, gfx12 ? 64u : 128u}) {
    bind_image(Image{.width = 4, .height = 1, .pitch = pitch, .rdna_format = 7});
    // The ordinary resource carries MIN_LOD_WARN in the linked scale bit range.
    cu_->write_sgpr(sb_ + 5, 1u << (gfx12 ? 5 : 8));
    for (float u : {0.5f, 1.0f, 1.37109375f, 1.5f, 2.25f, 3.3125f}) {
      write_coordinates(0, {std::bit_cast<uint32_t>(u), std::bit_cast<uint32_t>(0.5f), 0});
      uint32_t ordinary = 0;
      for (uint8_t flags : {0u, 1u, 2u, 3u}) {
        SCOPED_TRACE(testing::Message()
                     << "pitch=" << pitch << " u=" << u << " flags=" << unsigned(flags));
        constexpr uint32_t sentinel = 0xdeadbeef;
        cu_->write_vgpr(vb_ + 8, 0, sentinel);
        cu_->write_vgpr(vb_ + 9, 0, sentinel);
        rdna4::VsampleBuilderFields modern{};
        modern.dim = modern.dmask = modern.unorm = 1;
        modern.vdata = modern.samp = 8;
        modern.vaddr1 = 1;
        modern.tfe = flags & 1;
        modern.lwe = flags >> 1;
        rdna3_5::MimgBuilderFields legacy{};
        legacy.dim = legacy.dmask = legacy.unorm = 1;
        legacy.vdata = 8;
        legacy.ssamp = 2;
        legacy.tfe = flags & 1;
        legacy.lwe = flags >> 1;
        auto decoded =
            gfx12
                ? decoder->decode_window(rdna4::build_vsample(rdna4::kImageSampleLzVsample, modern))
                : decoder->decode_window(rdna3_5::build_mimg(rdna3_5::kImageSampleLzMimg, legacy));
        ASSERT_TRUE(decoded.succeeded());
        ASSERT_TRUE(cu_->execute_instruction(decoded.value().get(), *wf_).succeeded());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(decoded.value().release(), *wf_),
                  amdgpu::VmAccessOutcome::Complete);
        const uint32_t result = cu_->read_vgpr(vb_ + 8, 0);
        if (!flags)
          ordinary = result;
        EXPECT_EQ(result, ordinary);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), sentinel);
        if (u == 1.0f) {
          EXPECT_EQ(result, std::bit_cast<uint32_t>(0.5f));
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, R16UnormFiltersWithImplicitOrExplicitPitch) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP();
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  Image image;
  image.rdna_format = 7; // R16_UNORM.
  image.height = 2;
  memory_.write32(kBase, 0xffffffffu);
  memory_.write32(kBase + (gfx12 ? 128u : 256u), 0xffffffffu);
  wf_->set_exec(1);
  for (const uint32_t width : {1u, 2u}) {
    image.width = width;
    for (const bool linear : {false, true}) {
      for (const uint32_t pitch : {gfx12 ? 64u : 128u, 1u}) {
        SCOPED_TRACE(testing::Message()
                     << "width=" << width << " linear=" << linear << " pitch=" << pitch);
        image.pitch = pitch;
        bind_image(image);
        bind_sampler({.linear_xy = linear});
        write_coordinates(0, {std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(0.5f), 0});
        auto instruction = sample_instruction(SampleTestOperation::Zero);
        ASSERT_NE(instruction, nullptr);
        wf_->clear_instruction_execution_error();
        ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wf_).succeeded())
            << int(wf_->instruction_execution_error());
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(instruction.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 0)), 1.0f);
      }
    }
  }
}

TEST_P(Gfx11ImageFeedbackTest, TfeImplicitSamplesReadOnlyContributingCoordinatesAtFixedLod) {
  class Inputs final : public ExecutionPlugin {
  public:
    Inputs() : ExecutionPlugin("sampler_reads") {}
    void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wave, uint32_t reg, uint64_t lanes,
                               uint8_t) override {
      if (reg == wave->vgpr_alloc().base)
        read_lanes |= lanes;
    }
    uint64_t read_lanes = 0;
  };
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<Inputs>();
  auto *inputs = observer.get();
  ASSERT_TRUE(group->add(std::move(observer)));
  cu_->set_plugin_group(group);
  Image image;
  image.width = image.height = 16;
  image.pitch = 1;
  image.mip_levels = 2;
  image.last_level = 1;
  image.rdna_format = 13;
  const auto resource = rdna_descriptor(image, arch());
  for (unsigned i = 0; i < resource.size(); ++i)
    cu_->write_sgpr(sb_ + i, resource[i]);
  cu_->write_sgpr(sb_ + 8, 2 | (2 << 3));
  cu_->write_sgpr(sb_ + 9, 0xfffu << 12);
  for (unsigned lane = 0; lane < 4; ++lane)
    write_coordinates(lane, {0, 0, 0});
  auto decoder = Decoder::create(arch());
  auto &scoreboard = wf_->ensure_memory_wait_scoreboard();
  wf_->set_exec(1);
  for (const bool tfe : {false, true}) {
    for (const bool use_mip_filter : {false, true}) {
      SCOPED_TRACE(testing::Message() << "TFE=" << tfe << " mip-filter=" << use_mip_filter);
      cu_->write_sgpr(sb_ + 10, use_mip_filter ? 1u << 26 : 0);
      rdna3_5::MimgBuilderFields fields{};
      fields.dim = fields.dmask = 1;
      fields.vdata = 8;
      fields.ssamp = 2;
      fields.tfe = tfe;
      const auto words = rdna3_5::build_mimg(rdna3_5::kImageSampleMimg, fields);
      auto instruction = decoder->decode_window(words);
      ASSERT_TRUE(instruction.succeeded());
      unsigned warnings = 0;
      scoreboard.clear();
      const auto counter = amdgpu::WaitCounterKind::Ds;
      scoreboard.add({scoreboard.issue(counter), 0x100, 2, {RegClass::VGPR, 0, 1}, counter, 0xf});
      scoreboard.bind(0x200, &warnings,
                      [](void *ptr, const auto &) { ++*static_cast<unsigned *>(ptr); });
      scoreboard.check_instruction(*instruction.value(), *wf_);
      inputs->read_lanes = 0;
      ASSERT_TRUE(cu_->execute_instruction(instruction.value().get(), *wf_).succeeded());
      EXPECT_EQ(bool(inputs->read_lanes & 2), use_mip_filter);
      EXPECT_EQ(warnings, unsigned(bool(inputs->read_lanes & 2)));
    }
  }
}

TEST_P(ImageSamplerControlTest, SrgbChannelCountsShareNativePitchAndFiltering) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();

  wf_->set_exec(1);
  bind_sampler({.linear_xy = true});
  const uint32_t row_bytes = gfx12 ? 128 : 256;
  struct Format {
    uint32_t id, texel_bytes;
  };
  for (const Format format : {Format{1, 1}, Format{64, 1}, Format{65, 2}, Format{66, 4}}) {
    const uint32_t width = row_bytes / format.texel_bytes;
    for (uint32_t y = 0; y < 2; ++y) {
      const auto row = kBase + y * row_bytes;
      if (format.texel_bytes == 1) {
        memory_.write32(row, 0x0000c040u);
      } else if (format.texel_bytes == 2) {
        memory_.write32(row, 0x00c00040u);
      } else {
        memory_.write32(row, 0xff000040u);
        memory_.write32(row + 4, 0xff0000c0u);
      }
    }
    write_coordinates(0, {std::bit_cast<uint32_t>(1.0f / width), std::bit_cast<uint32_t>(0.5f), 0});
    uint32_t explicit_result = 0;
    for (const uint32_t pitch : {width - 1, 0u}) {
      const uint32_t extent = width - 1;
      const std::array<uint32_t, 8> descriptor{uint32_t(kBase >> 8),
                                               (format.id << (gfx12 ? 17 : 20)) | (extent << 30),
                                               (extent >> 2) | (1u << 14),
                                               (9u << 28) | 4u,
                                               pitch,
                                               0,
                                               0,
                                               0};
      for (unsigned word = 0; word < descriptor.size(); ++word)
        cu_->write_sgpr(sb_ + word, descriptor[word]);
      constexpr uint32_t sentinel = 0xabcdef01u;
      cu_->write_vgpr(vb_ + 8, 0, sentinel);
      const bool issued = sample(SampleTestOperation::Zero);
      const uint32_t result = cu_->read_vgpr(vb_ + 8, 0);
      std::printf("SAMPLE arch=%d format=%u pitch_field=%u row_bytes=%u issued=%d dest=%08x\n",
                  int(arch()), format.id, pitch, row_bytes, int(issued), result);
      SCOPED_TRACE(testing::Message() << "format=" << format.id << " pitch_field=" << pitch);
      EXPECT_TRUE(issued);
      EXPECT_NE(result, sentinel);
      if (pitch) {
        explicit_result = result;
      } else if (issued) {
        EXPECT_EQ(result, explicit_result);
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, MipChainPitchAdmissionIsIndependentOfFeedback) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  const bool gfx115 = arch() == ROCJITSU_CODE_ARCH_RDNA3_5;
  if (!gfx12 && !gfx115 && arch() != ROCJITSU_CODE_ARCH_RDNA3)
    GTEST_SKIP();
  wf_->set_exec(1);
  bind_sampler();
  write_coordinates(0,
                    std::bit_cast<std::array<uint32_t, 3>>(std::array{1.5f / 16, 1.5f / 8, 0.f}));
  for (const uint32_t pitch : {1u, 8u, 15u, 16u, 64u, 100u, 128u, 129u, 192u}) {
    bind_image({.width = 16, .height = 8, .pitch = pitch, .mip_levels = 2, .last_level = 1});
    // The fixture encodes pitch - 1. A zero field requests native packing.
    const bool valid = pitch == 1 || (pitch >= 16 && (!gfx115 || pitch % 64 == 0));
    const uint32_t width = pitch == 1 ? 16 : pitch;
    const uint32_t row = gfx12 ? 32 : 64;
    const uint64_t mip1_size = (((width + 1) / 2 + 63) / 64) * 64 * 4 * sizeof(uint32_t);
    const uint32_t row_pitch = ((width + row - 1) / row) * row;
    const uint32_t value = 0x12340000u + pitch;
    memory_.write32(kBase + mip1_size + (row_pitch + 1) * sizeof(uint32_t), value);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    for (bool tfe : {false, true}) {
      SCOPED_TRACE(testing::Message() << "pitch=" << pitch << " tfe=" << tfe);
      wf_->clear_instruction_execution_error();
      cu_->write_vgpr(vb_ + 8, 0, 0xdeadbeef);
      auto instruction = modern_sample_lz(false, 1, tfe);
      ASSERT_TRUE(instruction.succeeded());
      ASSERT_EQ(cu_->execute_instruction(instruction.value().get(), *wf_).succeeded(), valid);
      if (valid) {
        amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
        ASSERT_EQ(pipeline.issue(instruction.value().release(), *wf_),
                  amdgpu::VmAccessOutcome::Complete);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), value);
      } else {
        EXPECT_EQ(wf_->instruction_execution_error(),
                  amdgpu::InstructionExecutionError::UnsupportedOperandValue);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xdeadbeef);
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, ExtendedPitchSetsEveryMipRowAndLevelOffset) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP();
  Image image;
  image.width = 16;
  image.height = 8;
  image.rdna_format = 42;
  image.pitch = gfx12 ? 96 : 192;
  image.mip_levels = 5;
  image.last_level = 4;
  bind_image(image);
  bind_sampler({.mip = MipFilter::Point, .max_lod_fixed8 = 4u << 8});
  wf_->set_exec(1);

  // Physical gfx1100 and gfx1201 point samples of X=0 at the specified Y and LOD.
  struct Address {
    uint32_t level, y, gfx11, gfx12;
  };
  for (const auto &test :
       {Address{0, 0, 3072, 2048}, Address{0, 1, 3840, 2432}, Address{1, 0, 1024, 1024},
        Address{1, 1, 1536, 1280}, Address{2, 1, 768, 640}, Address{3, 0, 256, 256}}) {
    SCOPED_TRACE(testing::Message() << "level=" << test.level << " y=" << test.y);
    const uint32_t width = std::max(1u, image.width >> test.level);
    const uint32_t height = std::max(1u, image.height >> test.level);
    const uint64_t address = kBase + (gfx12 ? test.gfx12 : test.gfx11);
    memory_.write32(address, 0xffu);
    write_coordinates(0, {std::bit_cast<uint32_t>(0.5f / width),
                          std::bit_cast<uint32_t>((test.y + 0.5f) / height),
                          std::bit_cast<uint32_t>(float(test.level))});
    ASSERT_TRUE(sample(SampleTestOperation::Lod));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(1.0f));
    memory_.write32(address, 0);
  }
}

TEST_P(ImageSamplerControlTest, GatherPreservesFloatPayloadsThatPointSamplingCanonicalizes) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA omits gather forms";
  Image image;
  image.width = image.height = 2;
  bind_sample_image(image);
  bind_sampler({.normalized = false});
  wf_->set_exec(1);
  // Captured on gfx1100/gfx1201: gather preserves NaN payload and denormal bits.
  for (uint32_t payload : {0x7fc12345u, 0x7f800001u, 0xffc00001u, 1u, 0x80000001u, 0x007fffffu,
                           0x80000000u, 0x7f800000u}) {
    memory_.write32(kBase, payload);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(0.5f));
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    const bool nan = (payload & 0x7fffffffu) > 0x7f800000u;
    const bool denormal = (payload & 0x7f800000u) == 0;
    const uint32_t point = nan ? 0xffc00000u : denormal ? payload & 0x80000000u : payload;
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), point);
    cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(1.0f));
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(1.0f));
    ASSERT_TRUE(sample(SampleTestOperation::Gather));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 11, 0), payload);
  }
}

TEST_P(ImageSamplerControlTest, GatherReturnsAllFourTexelsInComponentOrder) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA omits gather forms";
  Image image;
  image.width = image.height = 2;
  bind_sample_image(image);
  bind_sampler({.normalized = false});
  wf_->set_exec(1);
  write_coordinates(0, {std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(1.0f), 0});
  for (uint32_t y = 0; y < 2; ++y)
    for (uint32_t x = 0; x < 2; ++x)
      memory_.write32(kBase + 256 * y + 4 * x, std::bit_cast<uint32_t>(float(1 + 2 * y + x)));

  // A row-major {1, 2, 3, 4} returned these F32 and D16 tuples on gfx1100 and gfx1201.
  ASSERT_TRUE(sample(SampleTestOperation::Gather));
  constexpr std::array<float, 4> expected{3, 4, 2, 1};
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + i, 0), std::bit_cast<uint32_t>(expected[i]));
  ASSERT_TRUE(sample(SampleTestOperation::Gather, true));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0x44004200u);
  EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x3c004000u);
}

TEST_P(ImageSamplerControlTest, LinearXyFilterBlendsVolumeSlicesWithZFilterNone) {
  Image image;
  image.type = amdgpu::kImage3d;
  image.width = image.depth = 2;
  dimension_override_ = amdgpu::kImageDim3d;
  bind_sample_image(image);
  bind_sampler({.linear_xy = true, .normalized = false});
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, std::bit_cast<uint32_t>(0.75f)});
  for (uint32_t x = 0; x < 2; ++x) {
    memory_.write32(kBase + 4 * x, 0);
    memory_.write32(kBase + 256 + 4 * x, std::bit_cast<uint32_t>(4.0f));
  }
  // gfx1100 and gfx1201 with ROCr's XY linear and Z_FILTER_NONE sampler
  // return 0.75 * slice(0) + 0.25 * slice(1) = 1 at this coordinate.
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(1.0f));
}

TEST_P(ImageSamplerControlTest, UnqualifiedLodControlsAreRejected) {
  const bool legacy = arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA1 ||
                      arch() == ROCJITSU_CODE_ARCH_RDNA2;
  for (bool resource_control : {false, true}) {
    if (!resource_control && !legacy)
      continue;
    Image image;
    image.min_lod = resource_control ? 1 : 0;
    bind_sample_image(image);
    bind_sampler({.mip = MipFilter::Point});
    if (!resource_control) {
      cu_->write_sgpr(sb_ + 10, cu_->read_sgpr(sb_ + 10) | (1u << 28));
    }
    wf_->set_exec(1);
    wf_->clear_instruction_execution_error();
    if (resource_control &&
        (arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2)) {
      ASSERT_TRUE(sample(SampleTestOperation::Zero));
      continue;
    }
    EXPECT_FALSE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
  }
  if (!amdgpu::has_gfx11_plus_image_operands(arch()))
    return;
  unsigned reports = 0;
  auto &waits = wf_->ensure_memory_wait_scoreboard();
  waits.bind(0x200, &reports, [](void *p, const auto &) { ++*static_cast<unsigned *>(p); });
  constexpr auto counter = amdgpu::WaitCounterKind::Load;
  constexpr uint32_t sentinel = 0xdeadbeef;
  for (uint32_t min_lod : {0u, 1u, 256u}) {
    const bool uav3d = !min_lod;
    if (uav3d && arch() != ROCJITSU_CODE_ARCH_RDNA4)
      continue;
    for (uint32_t levels : {1u, 2u}) {
      bind_sample_image({.width = 4,
                         .height = 4,
                         .mip_levels = levels,
                         .last_level = levels - 1,
                         .min_lod = min_lod});
      if (uav3d)
        cu_->write_sgpr(sb_ + 5, cu_->read_sgpr(sb_ + 5) | (1u << 4));
      bind_sampler();
      for (bool tfe : {false, true}) {
        for (uint64_t exec : {0u, 1u}) {
          SCOPED_TRACE(testing::Message()
                       << "min_lod=" << min_lod << " uav3d=" << uav3d << " levels=" << levels
                       << " tfe=" << tfe << " EXEC=" << exec);
          wf_->set_exec(exec);
          auto instruction = modern_sample_lz(false, 1, tfe);
          ASSERT_TRUE(instruction.succeeded());
          reports = 0;
          waits.clear();
          waits.add({waits.issue(counter), 0x100, 1, {RegClass::VGPR, 0, 1}, counter, 0xf});
          waits.check_instruction(*instruction.value(), *wf_);
          EXPECT_EQ(reports, 0u);
          waits.clear();
          wf_->clear_instruction_execution_error();
          cu_->write_vgpr(vb_ + 8, 0, sentinel);
          cu_->write_vgpr(vb_ + 9, 0, sentinel);
          EXPECT_TRUE(cu_->execute_instruction(instruction.value().get(), *wf_).failed());
          EXPECT_EQ(wf_->instruction_execution_error(),
                    amdgpu::InstructionExecutionError::UnimplementedInstruction);
          EXPECT_FALSE(instruction.value()->data());
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), sentinel);
          EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), sentinel);
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, OneDimensionalViewUsesZeroForTheMissingCoordinate) {
  if (arch_is_cdna_4_or_lower(arch()))
    GTEST_SKIP() << "GFX9 has no DIM operand";
  for (bool layered : {false, true}) {
    for (uint32_t layer = 0; layer < (layered ? 2u : 1u); ++layer) {
      Image image;
      image.width = 4;
      image.height = 2;
      image.type = layered ? amdgpu::kImage2dArray : amdgpu::kImage2d;
      image.depth = layered ? 2 : 1;
      dimension_override_ = layered ? amdgpu::kImageDim1dArray : amdgpu::kImageDim1d;
      bind_sample_image(image);
      // GFX12 array rows use 128-byte alignment, while allocation keeps
      // 256-byte rows for the layer stride (two rows = 512 bytes).
      const uint32_t row_bytes = layered && arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
      for (uint32_t i = 0; i < 4; ++i) {
        memory_.write32(kBase + layer * 512 + i * 4, std::bit_cast<uint32_t>(1000.0f * layer + i));
        memory_.write32(kBase + layer * 512 + row_bytes + i * 4,
                        std::bit_cast<uint32_t>(1000.0f * layer + 100.0f + i));
      }
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      wf_->set_exec(15);
      for (bool normalized : {false, true}) {
        for (bool linear : {false, true}) {
          for (Clamp wrap : {Clamp::Repeat, Clamp::Mirror, Clamp::Edge, Clamp::Border}) {
            bind_sampler({.clamp_y = wrap, .linear_xy = linear, .normalized = normalized});
            for (uint32_t lane = 0; lane < 4; ++lane) {
              const float x = lane + (linear ? 0.25f : 0.5f);
              cu_->write_vgpr(vb_, lane, std::bit_cast<uint32_t>(normalized ? x / 4 : x));
              // The instruction has no Y input; a following live VGPR is irrelevant.
              cu_->write_vgpr(vb_ + 1, lane,
                              std::bit_cast<uint32_t>(layered ? float(layer) : 1.5f));
              cu_->write_vgpr(vb_ + 2, lane, std::bit_cast<uint32_t>(1.5f));
            }
            for (bool tfe : {false, true}) {
              if (tfe && !amdgpu::has_gfx11_plus_image_operands(arch()))
                continue;
              if (amdgpu::has_gfx11_plus_image_operands(arch())) {
                ASSERT_NO_FATAL_FAILURE(
                    issue_modern_sample_lz(!normalized, 1, tfe, *dimension_override_));
              } else {
                ASSERT_TRUE(sample(SampleTestOperation::Zero));
              }
              for (uint32_t lane = 0; lane < 4; ++lane) {
                float expected = 1000.0f * layer + (linear && lane ? lane - 0.25f : float(lane));
                if (linear && wrap == Clamp::Border)
                  expected *= 0.5f; // Half the footprint reads transparent black.
                else if (linear && normalized && wrap == Clamp::Repeat)
                  expected += 50.0f; // Half reads the last row, whose values are 100 higher.
                EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), std::bit_cast<uint32_t>(expected));
              }
            }
          }
        }
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, GenerationSpecificUnsupportedSamplerControlsPreserveDestination) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  if (arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "GFX12 has none of these controls";
  wf_->set_exec(1);
  for (bool filter_precision : {false, true}) {
    if (filter_precision && !gfx9)
      continue;
    Image image;
    image.width = image.height = 2;
    bind_sampler({.linear_xy = filter_precision, .mip = gfx9 ? MipFilter::None : MipFilter::Point});
    if (gfx9 && !filter_precision) {
      image.data_format = kDfmt8888;
      image.num_format = 9; // GFX9 sRGB.
      bind_image(image);
      memory_.write32(kBase, 0x80402020);
      cu_->write_sgpr(sb_ + 11, 1u << 12); // SKIP_DEGAMMA
    } else {
      bind_sample_image(image);
      if (filter_precision)
        cu_->write_sgpr(sb_ + 10, cu_->read_sgpr(sb_ + 10) | (1u << 30)); // FILTER_PREC_FIX
      else {
        cu_->write_sgpr(sb_ + 10, cu_->read_sgpr(sb_ + 10) | (1u << 31)); // DERIV_ADJUST_EN
        cu_->write_sgpr(sb_ + 11, 0x01234567);
      }
    }
    for (uint32_t reg = 0; reg < 6; ++reg)
      cu_->write_vgpr(vb_ + reg, 0, std::bit_cast<uint32_t>(0.25f));
    cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
    wf_->clear_instruction_execution_error();
    const SampleTestOperation operation =
        gfx9 ? SampleTestOperation::Zero : SampleTestOperation::Gradients;
    EXPECT_FALSE(sample(operation));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
    EXPECT_TRUE(wf_->wait_counters().empty());
  }
}

TEST_P(ImageSamplerControlTest, UnqualifiedSamplerFlagsPreserveDestination) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  bind_sample_image(Image{});
  bind_sampler();
  wf_->set_exec(1);
  memory_.write32(kBase, std::bit_cast<uint32_t>(0.75f));
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(0.75f));

  struct Flag {
    uint32_t word, bit;
    bool present;
  };
  const Flag flags[] = {
      {0, 31, gfx9}, // COMPAT_MODE
      {2, gfx9 ? 29u : 30u,
       !gfx12 && !amdgpu::has_gfx11_plus_image_operands(arch())}, // BLEND_ZERO_PRT
      {2, gfx9 ? 31u : 29u, true}};                               // ANISO_OVERRIDE
  for (const auto &flag : flags) {
    if (!flag.present)
      continue;
    SCOPED_TRACE(testing::Message() << flag.word << ':' << flag.bit);
    bind_sampler();
    cu_->write_sgpr(sb_ + 8 + flag.word, cu_->read_sgpr(sb_ + 8 + flag.word) | (1u << flag.bit));
    cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
    wf_->clear_instruction_execution_error();
    EXPECT_FALSE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
    EXPECT_TRUE(wf_->wait_counters().empty());
  }
}

TEST_P(ImageSamplerControlTest, UnbackedSamplerChecksTheNonNullResourceFirst) {
  const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
  const bool modern = amdgpu::has_gfx11_plus_image_operands(arch());
  wf_->set_exec(1);
  bind_sampler();
  memory_.write32(kBase, std::bit_cast<uint32_t>(0.75f));
  for (uint32_t type : {amdgpu::kImage2d, amdgpu::kImageCube, amdgpu::kImage2dMsaa}) {
    Image image;
    image.type = type;
    if (type == amdgpu::kImageCube) {
      image.depth = 6;
      write_coordinates(0, {std::bit_cast<uint32_t>(1.5f), std::bit_cast<uint32_t>(1.5f), 0});
    }
    bind_sample_image(image);
    dimension_override_ = type - amdgpu::kImage1d;
    for (uint32_t sampler : {8u, 104u}) {
      SCOPED_TRACE(testing::Message() << type << ':' << sampler);
      cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
      wf_->clear_instruction_execution_error();
      if (type == amdgpu::kImage2d || ((gfx10 || modern) && type == amdgpu::kImageCube)) {
        ASSERT_TRUE(sample(SampleTestOperation::Zero, false, sampler));
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), sampler == 8 ? std::bit_cast<uint32_t>(0.75f) : 0u);
      } else {
        EXPECT_FALSE(sample(SampleTestOperation::Zero, false, sampler));
        EXPECT_EQ(wf_->instruction_execution_error(),
                  (gfx10 && type == amdgpu::kImage2dMsaa) || modern
                      ? amdgpu::InstructionExecutionError::UnsupportedOperandValue
                      : amdgpu::InstructionExecutionError::UnimplementedInstruction);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
        EXPECT_TRUE(wf_->wait_counters().empty());
      }
    }
  }
}

TEST_P(ImageSamplerControlTest, LinkedResourcesStopBeforeSampleLoadOrStore) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  bind_sample_image(Image{});
  bind_sampler();
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  constexpr uint32_t texel = std::bit_cast<uint32_t>(0.75f);
  constexpr uint32_t sentinel = 0xabcdef01;
  memory_.write32(kBase, texel);
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  ASSERT_EQ(cu_->read_vgpr(vb_ + 8, 0), texel);
  // Word 5 bit 24 selects a linked resource on GFX10/11/12. On GFX9 it
  // belongs to the metadata address, unused by this uncompressed image.
  cu_->write_sgpr(sb_ + 5, cu_->read_sgpr(sb_ + 5) | (1u << 24));
  enum class Operation { Sample, Load, Store };
  for (Operation op : {Operation::Sample, Operation::Load, Operation::Store}) {
    SCOPED_TRACE(static_cast<int>(op));
    wf_->clear_instruction_execution_error();
    cu_->write_vgpr(vb_ + 8, 0, sentinel);
    const bool issued = op == Operation::Sample ? sample(SampleTestOperation::Zero)
                                                : decoded_issue(op == Operation::Load, 2);
    EXPECT_EQ(issued, gfx9);
    if (!gfx9) {
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), gfx9 && op != Operation::Store ? texel : sentinel);
    EXPECT_EQ(memory_.read32(kBase), gfx9 && op == Operation::Store ? sentinel : texel);
    EXPECT_TRUE(wf_->wait_counters().empty());
  }
}

TEST_P(ImageSamplerControlTest, InvalidFormatCompletesBeforeResourceTypeQualification) {
  Image image;
  image.type = amdgpu::kImageCube;
  image.data_format = image.rdna_format = 0; // INVALID
  bind_image(image);
  bind_sampler();
  wf_->set_exec(1);
  constexpr uint32_t texel = 0x12345678;
  constexpr uint32_t sentinel = 0xdeadbeef;
  memory_.write32(kBase, texel);
  cu_->write_vgpr(vb_ + 8, 0, sentinel);
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
  cu_->write_vgpr(vb_ + 8, 0, sentinel);
  ASSERT_TRUE(decoded_issue(true, 2));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0u);
  cu_->write_vgpr(vb_ + 8, 0, sentinel);
  ASSERT_TRUE(decoded_issue(false, 2));
  EXPECT_EQ(memory_.read32(kBase), texel);
}

TEST_P(ImageSamplerControlTest, SrgbConversionPrecedesSpatialAndMipFiltering) {
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx10 = arch() == ROCJITSU_CODE_ARCH_RDNA1 || arch() == ROCJITSU_CODE_ARCH_RDNA2;
  wf_->set_exec(15);
  for (uint32_t lane = 0; lane < 4; ++lane)
    write_coordinates(lane, {std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(0.5f),
                             std::bit_cast<uint32_t>(0.5f)});
  // gfx1100 and gfx1201 returned 0x1.284p-2 for both filters, versus
  // 0x1.010102p-1 with R8_UNORM. Earlier ISAs use the IEC sRGB transfer
  // of 64/255 and 192/255 followed by their arithmetic mean.
  const float expected = gfx9 || gfx10 ? 0.289192f : 0x1.284p-2f;
  for (bool mip : {false, true}) {
    SCOPED_TRACE(mip);
    Image image;
    image.width = image.height = 2;
    image.data_format = 1; // R8
    image.num_format = 9;  // sRGB
    image.rdna_format = gfx10 ? 128 : 64;
    image.pitch = mip && arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
    image.mip_levels = mip ? 2 : 1;
    image.last_level = image.mip_levels - 1;
    bind_image(image);
    // CDNA2 exposes the base SAMPLE form. Its MIN_LOD clamp supplies the
    // same fractional level as the explicit LOD operand on the other ISAs.
    const bool implicit_mip = mip && arch() == ROCJITSU_CODE_ARCH_CDNA2;
    bind_sampler({.linear_xy = !mip,
                  .mip = mip ? MipFilter::Linear : MipFilter::None,
                  .min_lod_fixed8 = implicit_mip ? 128u : 0u});
    const uint64_t base = kBase + (mip && !gfx9 ? 256 : 0);
    const uint32_t pitch = image.pitch;
    for (uint32_t y = 0; y < 2; ++y)
      memory_.write32(base + y * pitch, 64 | ((mip ? 64u : 192u) << 8));
    if (mip)
      memory_.write32(kBase + (gfx9 ? 512 : 0), 192);
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    ASSERT_TRUE(
        sample(mip && !implicit_mip ? SampleTestOperation::Lod : SampleTestOperation::Zero));
    EXPECT_NEAR(std::bit_cast<float>(cu_->read_vgpr(vb_ + 8, 0)), expected, 0.0025f);
  }
}

TEST_P(ImageSamplerControlTest, D16FloatFilteringMatchesGfx11AndGfx12Captures) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "D16 filtering captured on gfx1100 and gfx1201";
  // R32_FLOAT, blend exactly halfway between 1 and each high texel.
  // Physical gfx1100 and gfx1201 agree for spatial and mip filtering.
  // In particular 1 + 3/4096 returns 0x3c00, distinguishing truncation
  // from round-to-nearest-even (0x3c01).
  struct Case {
    float high;
    uint16_t expected;
  };
  constexpr Case cases[] = {
      {0x1.004p+0f, 0x3c00}, {0x1.006p+0f, 0x3c00}, {0x1.008p+0f, 0x3c01}, {0x1.00cp+0f, 0x3c01}};
  const uint32_t row_pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
  wf_->set_exec(1);
  write_coordinates(0, {std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(0.5f),
                        std::bit_cast<uint32_t>(0.5f)});
  for (bool mip : {false, true}) {
    Image image;
    image.width = image.height = 2;
    image.pitch = row_pitch / sizeof(float);
    image.mip_levels = mip ? 2 : 1;
    image.last_level = mip ? 1 : 0;
    bind_sample_image(image);
    bind_sampler({.linear_xy = !mip, .mip = mip ? MipFilter::Linear : MipFilter::None});
    for (const Case &test : cases) {
      SCOPED_TRACE(testing::Message() << mip << ':' << test.high);
      const uint64_t base_level = kBase + (mip ? 256 : 0);
      for (uint32_t row = 0; row < 2; ++row) {
        memory_.write32(base_level + row * row_pitch, std::bit_cast<uint32_t>(1.0f));
        memory_.write32(base_level + row * row_pitch + 4,
                        std::bit_cast<uint32_t>(mip ? 1.0f : test.high));
      }
      if (mip)
        memory_.write32(kBase, std::bit_cast<uint32_t>(test.high));
      cu_->l1_vector().invalidate_all();
      l2_.invalidate_all();
      cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
      ASSERT_TRUE(sample(mip ? SampleTestOperation::Lod : SampleTestOperation::Zero, true));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0) & 0xffffu, test.expected);
    }
  }
}

TEST_P(ImageSamplerControlTest, CornerAndPerformanceControlsQualifyTheSample) {
  struct Case {
    const char *name;
    Image image;
    Sampler sampler;
    bool accepted;
  };
  const Case cases[] = {
      {"CORNER_SAMPLES", {.corner_samples = true}, {}, false},
      {"PERF_MOD", {.perf_mod = 1}, {}, true},
      // GFX12 splits PERF_MIP over words 2 and 3. This value sets bits in both.
      {"PERF_MIP", {}, {.perf_mip = 11}, true},
      {"PERF_MOD + PERF_MIP",
       {.perf_mod = 1},
       {.perf_mip = 8},
       amdgpu::has_gfx11_plus_image_operands(arch())},
      {"Volume PERF_Z", {.depth = 2, .type = amdgpu::kImage3d}, {.perf_z = 15}, true},
      {"Volume PERF_MOD + PERF_Z",
       {.depth = 2, .type = amdgpu::kImage3d, .perf_mod = 1},
       {.perf_z = 15},
       false},
  };
  constexpr uint32_t texel = std::bit_cast<uint32_t>(0.75f);
  constexpr uint32_t sentinel = 0xabcdef01;
  memory_.write32(kBase, texel);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (const Case &test : cases) {
    if (test.image.corner_samples && arch_is_cdna_4_or_lower(arch()))
      continue; // GFX9 has no CORNER_SAMPLES.
    SCOPED_TRACE(test.name);
    dimension_override_ =
        test.image.type == amdgpu::kImage3d ? amdgpu::kImageDim3d : amdgpu::kImageDim2d;
    bind_sample_image(test.image);
    bind_sampler(test.sampler);
    cu_->write_vgpr(vb_ + 8, 0, sentinel);
    wf_->clear_instruction_execution_error();
    EXPECT_EQ(sample(SampleTestOperation::Zero), test.accepted);
    if (!test.accepted) {
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), test.accepted ? texel : sentinel);
  }
  if (arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
      arch() != ROCJITSU_CODE_ARCH_RDNA4)
    return;
  const uint32_t swizzle = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 3 : 27;
  bind_image(Image{.width = 4,
                   .height = 4,
                   .pitch = 1,
                   .swizzle_mode = swizzle,
                   .rdna_format = 13,
                   .corner_samples = true});
  for (bool linear : {false, true}) {
    bind_sampler({.linear_xy = linear});
    for (bool tfe : {false, true}) {
      SCOPED_TRACE(testing::Message() << "tiled R16_FLOAT linear=" << linear << " tfe=" << tfe);
      wf_->clear_instruction_execution_error();
      cu_->write_vgpr(vb_ + 8, 0, sentinel);
      cu_->write_vgpr(vb_ + 9, 0, sentinel);
      auto inst = modern_sample_lz(false, 1, tfe);
      ASSERT_TRUE(inst.succeeded());
      EXPECT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).failed());
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_FALSE(inst.value()->data());
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), sentinel);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), sentinel);
    }
  }
}

TEST_P(ImageSamplerControlTest, ScaledViewsStopBeforeSampling) {
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  if (!gfx12 && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5)
    GTEST_SKIP() << "GFX11 and GFX12 expose separate resource scales";
  Image image;
  image.type = amdgpu::kImage3d;
  image.depth = 2;
  dimension_override_ = amdgpu::kImageDim3d;
  wf_->set_exec(1);
  for (uint32_t bit_index = gfx12 ? 4 : 8; bit_index <= 19; ++bit_index) {
    SCOPED_TRACE(bit_index);
    bind_sample_image(image);
    bind_sampler();
    const uint32_t linked = gfx12 && bit_index == 4 ? 0u : 1u << 24;
    cu_->write_sgpr(sb_ + 5, cu_->read_sgpr(sb_ + 5) | linked | (1u << bit_index));
    for (SampleTestOperation operation :
         {SampleTestOperation::Zero, SampleTestOperation::Lod, SampleTestOperation::Gradients}) {
      SCOPED_TRACE(static_cast<uint32_t>(operation));
      memory_.write32(kBase, 0x12345678);
      cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
      wf_->clear_instruction_execution_error();
      EXPECT_FALSE(sample(operation));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
      EXPECT_EQ(memory_.read32(kBase), 0x12345678u);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
      EXPECT_TRUE(wf_->wait_counters().empty());
    }
  }
}

TEST_P(ImageSamplerControlTest, TiledSamplerBaseXorIsRejected) {
  Image image;
  image.width = image.height = 64;
  image.swizzle_mode = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : 22;
  bind_sample_image(image);
  bind_sampler();
  cu_->write_sgpr(sb_, (kBase + 256) >> 8);
  wf_->set_exec(1);
  write_coordinates(0, {0, 0, 0});
  for (bool tfe : {false, true}) {
    if (tfe && arch() != ROCJITSU_CODE_ARCH_RDNA3 && arch() != ROCJITSU_CODE_ARCH_RDNA3_5 &&
        arch() != ROCJITSU_CODE_ARCH_RDNA4)
      continue;
    SCOPED_TRACE(tfe);
    texel_fault_ = tfe;
    wf_->clear_instruction_execution_error();
    cu_->write_vgpr(vb_ + 8, 0, 0xabcdef01);
    cu_->write_vgpr(vb_ + 9, 0, 0x87654321);
    EXPECT_FALSE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(wf_->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnimplementedInstruction);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 0xabcdef01u);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 9, 0), 0x87654321u);
  }
}

TEST(ImageFootprintTest, CapturedFractionalLodCoversLogSegmentsAndVolumeProjections) {
  struct Case {
    std::array<double, 3> dx, dy;
    std::array<uint32_t, 3> extents;
    uint32_t dimensions;
    double expected;
  };
  // Independent matching gfx1100/gfx1201 captures. Constant-valued mip levels
  // and linear mip filtering expose the sampler's fractional LOD directly.
  const Case cases[] = {
      // This fractional LOD distinguishes slope 129 from 128 in the fourth segment.
      {{0x1.955d52p-5, 0x1.f7ce2ap-5, 0},
       {0x1.f82884p-5, -0x1.312724p-4, 0},
       {37, 19, 1},
       2,
       0x1.91p+0},
      {{0x1.019d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1p-7},
      {{0x1.219d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.7p-3},
      {{0x1.419d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.5p-2},
      {{0x1.619d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.dcp-2},
      {{0x1.819d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.2ep-1},
      {{0x1.a19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.6ap-1},
      {{0x1.c19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.ap-1},
      {{0x1.e19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.d2p-1},
      {{0x1.019d000000000p-8, 0x1.019d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.02p-1},
      {{0x1.219d000000000p-8, 0x1.219d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.58p-1},
      {{0x1.419d000000000p-8, 0x1.419d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.a6p-1},
      {{0x1.619d000000000p-8, 0x1.619d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.ecp-1},
      {{0x1.819d000000000p-8, 0x1.819d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.16p+0},
      {{0x1.a19d000000000p-8, 0x1.a19d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.33p+0},
      {{0x1.c19d000000000p-8, 0x1.c19d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.4fp+0},
      {{0x1.e19d000000000p-8, 0x1.e19d000000000p-6, 0x0.0p+0},
       {0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.68p+0},
      {{0x1.019d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.019d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.02p-1},
      {{0x1.219d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.219d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.58p-1},
      {{0x1.419d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.419d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.a6p-1},
      {{0x1.619d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.619d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.ecp-1},
      {{0x1.819d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.819d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.16p+0},
      {{0x1.a19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.a19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.33p+0},
      {{0x1.c19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.c19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.4fp+0},
      {{0x1.e19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {0x1.e19d000000000p-8, 0x0.0p+0, 0x0.0p+0},
       {256, 64, 1},
       2,
       0x1.68p+0},
      {{-0x1.5bf1f8p-6, 0x1.208a3ep-6, -0x1.e9ea3ep-5},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{-0x1.e8bfccp-6, 0x1.954decp-6, -0x1.5815d2p-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {-0x1.bd65f2p-8, 0x1.a5ffcp-5, -0x1.1798d2p-5},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {-0x1.38d1c8p-7, 0x1.2862a6p-4, -0x1.88be08p-5},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{-0x1.26d1p-7, 0x1.fc4bc6p-7, -0x1.1ef782p-5},
       {-0x1.11b9dcp-7, 0x1.376612p-5, -0x1.12e478p-4},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{-0x1.9e1ee4p-7, 0x1.64febp-6, -0x1.931838p-5},
       {-0x1.807edcp-7, 0x1.b569d4p-5, -0x1.822252p-4},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{0x1.adaf82p-9, 0x1.306cf8p-5, -0x1.2bf50ep-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{0x1.2dc8aep-8, 0x1.ab9e5ap-5, -0x1.a55788p-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {-0x1.7487acp-8, -0x1.a5a39ap-5, 0x1.2b5c58p-5},
       {37, 19, 11},
       3,
       0x1.dp-4},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {-0x1.05a434p-7, -0x1.2821eep-4, 0x1.a48106p-5},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{0x1.13c47cp-11, -0x1.4ec31p-5, 0x1.1662e8p-4},
       {-0x1.4a8bd8p-6, 0x1.3f30aep-7, 0x1.429284p-5},
       {37, 19, 11},
       3,
       0x1.bp-4},
      {{0x1.835cfcp-11, -0x1.d63b2cp-5, 0x1.870ab8p-4},
       {-0x1.d04f3ap-6, 0x1.c05b9ap-7, 0x1.c51bd2p-5},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{0x1.f69754p-7, 0x1.28e23ep-7, 0x1.4f38eep-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{0x1.60fcf4p-6, 0x1.a1065p-7, 0x1.d6e0bep-4},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {0x1.69a6c8p-11, -0x1.d06f0ep-5, 0x1.9a7648p-9},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {0x1.fc008ep-11, -0x1.46305ap-4, 0x1.204852p-8},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{-0x1.13823ap-9, 0x1.4af7a6p-5, -0x1.f45ec6p-5},
       {0x1.daa314p-7, -0x1.dfee1ep-6, -0x1.17eee8p-4},
       {37, 19, 11},
       3,
       0x1.bp-4},
      {{-0x1.82ffe8p-9, 0x1.d0e6a6p-5, -0x1.5f6da4p-4},
       {0x1.4d5adcp-6, -0x1.51128ep-5, -0x1.8936f4p-4},
       {37, 19, 11},
       3,
       0x1.3p-1},
      {{-0x1.e410d6p-7, -0x1.90339ap-5, -0x1.feeacep-9},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.dp-4},
      {{-0x1.53fa24p-6, -0x1.191384p-4, -0x1.66d5fap-8},
       {0x0p+0, 0x0p+0, 0x0p+0},
       {37, 19, 11},
       3,
       0x1.34p-1},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {0x1.a609ap-7, 0x1.7b5c7p-12, -0x1.680f6ep-4},
       {37, 19, 11},
       3,
       0x1.bp-4},
      {{0x0p+0, 0x0p+0, 0x0p+0},
       {0x1.286994p-6, 0x1.0a706ep-11, -0x1.f9c458p-4},
       {37, 19, 11},
       3,
       0x1.32p-1},
      {{-0x1.5443bep-6, 0x1.b3ee3p-11, 0x1.c54ep-8},
       {-0x1.25e2ep-6, 0x1.067ad4p-5, -0x1.b94152p-5},
       {37, 19, 11},
       3,
       0x1.cp-4},
      {{-0x1.ddf5e8p-6, 0x1.322b7cp-10, 0x1.3e5f5ap-7},
       {-0x1.9cd066p-6, 0x1.70b2c6p-5, -0x1.35e8dep-4},
       {37, 19, 11},
       3,
       0x1.32p-1},
  };
  for (const auto &test : cases) {
    const double lod = amdgpu::image_gradient_lod(test.dx, test.dy, test.extents, test.dimensions);
    EXPECT_EQ(std::clamp(lod, 0.0, 5.0), test.expected);
  }
}

TEST_P(ImageSamplerControlTest, DecodedGradientsSelectCapturedBoundaryMips) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA omits explicit gradient forms";
  Image image;
  image.width = 37;
  image.height = 19;
  image.mip_levels = 6;
  image.last_level = image.mip_levels - 1;
  bind_sample_image(image);
  // Independent linear mip-atlas offsets for a 37x19 R32 allocation.
  // Storage rounds mip extents upward; the accessible view still rounds down.
  const std::array<uint32_t, 6> offsets =
      arch_is_cdna_4_or_lower(arch()) ? std::array<uint32_t, 6>{0, 4864, 7424, 8704, 9472, 9984}
                                      : std::array<uint32_t, 6>{5376, 2816, 1536, 768, 256, 0};
  const uint32_t pitch = arch() == ROCJITSU_CODE_ARCH_RDNA4 ? 128 : 256;
  for (uint32_t mip = 0; mip < 6; ++mip)
    for (uint32_t y = 0; y < std::max(1u, 19u >> mip); ++y)
      for (uint32_t x = 0; x < std::max(1u, 37u >> mip); ++x)
        memory_.write32(kBase + offsets[mip] + y * pitch + x * 4,
                        std::bit_cast<uint32_t>(float(mip)));
  struct Case {
    std::array<float, 4> gradients;
    float expected;
    bool linear_mip = false;
  };
  // Off-axis NPOT boundary captures; each gradient prefix is decoded and executed.
  const Case cases[] = {
      {{-0x1.85f86ap-6f, 0x1.4363e2p-6f, -0x1.467c04p-6f, 0x1.4f5242p-5f}, 0x0p+0f},
      {{-0x1.8a4898p-6f, 0x1.46f78cp-6f, -0x1.4a187p-6f, 0x1.5307b4p-5f}, 0x1p+0f},
      {{-0x1.85f86ap-5f, 0x1.4363e2p-5f, -0x1.467c04p-5f, 0x1.4f5242p-4f}, 0x1p+0f},
      {{-0x1.8a4898p-5f, 0x1.46f78cp-5f, -0x1.4a187p-5f, 0x1.5307b4p-4f}, 0x1p+1f},
      {{-0x1.85f86ap-4f, 0x1.4363e2p-4f, -0x1.467c04p-4f, 0x1.4f5242p-3f}, 0x1p+1f},
      {{-0x1.8a4898p-4f, 0x1.46f78cp-4f, -0x1.4a187p-4f, 0x1.5307b4p-3f}, 0x1.8p+1f},
      {{-0x1.85f86ap-3f, 0x1.4363e2p-3f, -0x1.467c04p-3f, 0x1.4f5242p-2f}, 0x1.8p+1f},
      {{-0x1.8a4898p-3f, 0x1.46f78cp-3f, -0x1.4a187p-3f, 0x1.5307b4p-2f}, 0x1p+2f},
      {{-0x1.ff83bep-8f, -0x1.8fff16p-9f, 0x1.6f4b42p-6f, -0x1.e72046p-5f}, 0x0p+0f},
      {{-0x1.02960ap-7f, -0x1.946ba8p-9f, 0x1.735b3ap-6f, -0x1.ec838ep-5f}, 0x1p+0f},
      {{-0x1.ff83bep-7f, -0x1.8fff16p-8f, 0x1.6f4b42p-5f, -0x1.e72046p-4f}, 0x1p+0f},
      {{-0x1.02960ap-6f, -0x1.946ba8p-8f, 0x1.735b3ap-5f, -0x1.ec838ep-4f}, 0x1p+1f},
      {{-0x1.ff83bep-6f, -0x1.8fff16p-7f, 0x1.6f4b42p-4f, -0x1.e72046p-3f}, 0x1p+1f},
      {{-0x1.02960ap-5f, -0x1.946ba8p-7f, 0x1.735b3ap-4f, -0x1.ec838ep-3f}, 0x1.8p+1f},
      {{-0x1.ff83bep-5f, -0x1.8fff16p-6f, 0x1.6f4b42p-3f, -0x1.e72046p-2f}, 0x1.8p+1f},
      {{-0x1.02960ap-4f, -0x1.946ba8p-6f, 0x1.735b3ap-3f, -0x1.ec838ep-2f}, 0x1p+2f},
      {{-0x1.982114p-7f, 0x1.46b13ap-7f, 0x1.edf4c2p-11f, -0x1.2bd04ap-4f}, 0x0p+0f},
      {{-0x1.9ca4acp-7f, 0x1.4a4e3ep-7f, 0x1.f36b5ep-11f, -0x1.2f2132p-4f}, 0x1p+0f},
      {{-0x1.982114p-6f, 0x1.46b13ap-6f, 0x1.edf4c2p-10f, -0x1.2bd04ap-3f}, 0x1p+0f},
      {{-0x1.9ca4acp-6f, 0x1.4a4e3ep-6f, 0x1.f36b5ep-10f, -0x1.2f2132p-3f}, 0x1p+1f},
      {{-0x1.982114p-5f, 0x1.46b13ap-5f, 0x1.edf4c2p-9f, -0x1.2bd04ap-2f}, 0x1p+1f},
      {{-0x1.9ca4acp-5f, 0x1.4a4e3ep-5f, 0x1.f36b5ep-9f, -0x1.2f2132p-2f}, 0x1.8p+1f},
      {{-0x1.982114p-4f, 0x1.46b13ap-4f, 0x1.edf4c2p-8f, -0x1.2bd04ap-1f}, 0x1.8p+1f},
      {{-0x1.9ca4acp-4f, 0x1.4a4e3ep-4f, 0x1.f36b5ep-8f, -0x1.2f2132p-1f}, 0x1p+2f},
      {{0x1.955d52p-5f, 0x1.f7ce2ap-5f, 0x1.f82884p-5f, -0x1.312724p-4f}, 0x1.91p+0f, true},
  };
  wf_->set_exec(1);
  for (const auto &test : cases) {
    // Constant-valued mips and linear blending expose the captured fractional LOD.
    bind_sampler(
        {.mip = test.linear_mip ? MipFilter::Linear : MipFilter::Point, .max_lod_fixed8 = 5 * 256});
    for (uint32_t word = 0; word < 4; ++word)
      cu_->write_vgpr(vb_ + word, 0, std::bit_cast<uint32_t>(test.gradients[word]));
    cu_->write_vgpr(vb_ + 4, 0, std::bit_cast<uint32_t>(0.5f));
    cu_->write_vgpr(vb_ + 5, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Gradients));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(test.expected));
  }
}

TEST_P(ImageSamplerControlTest, Gfx10SelectsTheXyFilterAfterLodClamping) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA1 && arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP();
  const auto bits = [](float value) { return std::bit_cast<uint32_t>(value); };
  bind_sample_image({.width = 8, .height = 8, .pitch = 1, .mip_levels = 4, .last_level = 3});
  for (unsigned i = 0; i < 1024; ++i)
    memory_.write32(kBase + 4 * i, bits(float(i)));
  wf_->set_exec(1);
  auto decoder = Decoder::create(arch());
  const auto check = [&](uint16_t opcode, float gradient, uint32_t minimum, uint32_t maximum,
                         float expected) {
    SCOPED_TRACE(testing::Message() << opcode << " gradient=" << gradient << " sampler=[" << minimum
                                    << "," << maximum << "]");
    bind_sampler({.mip = MipFilter::Point, .min_lod_fixed8 = minimum, .max_lod_fixed8 = maximum});
    cu_->write_sgpr(sb_ + 10, (1u << 26) | (1u << 22)); // Point MAG, linear MIN.
    const std::array<float, 7> coordinates{gradient, 0, 0, gradient, 0.3f, 0.3f, 0};
    for (unsigned i = 0; i < coordinates.size(); ++i)
      cu_->write_vgpr(vb_ + i, 0, bits(coordinates[i]));
    const auto words = rdna1::build_mimg(opcode, {.dim = 1, .dmask = 1, .vdata = 8, .ssamp = 2});
    auto inst = decoder->decode_window(words);
    ASSERT_TRUE(inst.succeeded());
    ASSERT_TRUE(cu_->execute_instruction(inst.value().get(), *wf_).succeeded());
    amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
    ASSERT_EQ(pipeline.issue(inst.value().release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), bits(expected));
  };
  constexpr auto sample = rdna1::kImageSampleDMimg;
  check(sample, 0.125f, 0, 3 * 256, 578.0f);
  check(sample, 0.25f, 0, 3 * 256, 237.44921875f);
  check(sample, 0.125f, 256, 3 * 256, 237.44921875f);
  check(sample, 0.25f, 0, 0, 578.0f);
}

TEST_P(ImageSamplerControlTest, ExplicitLodRoundsBeforeBiasAndSaturatesInfiniteInputs) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA and installed SDK omit explicit-LOD forms";
  Image image;
  image.width = image.height = 16;
  image.mip_levels = 4;
  image.last_level = image.mip_levels - 1;
  bind_sample_image(image);
  const bool gfx9 = arch_is_cdna_4_or_lower(arch());
  const bool gfx12 = arch() == ROCJITSU_CODE_ARCH_RDNA4;
  // Independent linear mip-atlas offsets; each mip contains its level number.
  const std::array<uint32_t, 4> offsets = gfx9 ? std::array<uint32_t, 4>{0, 4096, 6144, 7168}
                                               : std::array<uint32_t, 4>{3584, 1536, 512, 0};
  for (uint32_t level = 0; level < 4; ++level)
    for (uint32_t y = 0; y < (16u >> level); ++y)
      for (uint32_t x = 0; x < (16u >> level); ++x)
        memory_.write32(kBase + offsets[level] + y * (gfx12 ? 128 : 256) + x * 4,
                        std::bit_cast<uint32_t>(static_cast<float>(level)));
  struct Case {
    bool zero;
    float lod;
    int32_t bias;
    uint32_t minimum;
    float expected;
  };
  // Exact captures on gfx1100/gfx1201. Odd bias distinguishes conversion before
  // bias from a superficially similar nearest-even conversion after bias.
  const Case cases[] = {{true, 0, 128, 0, 0.5f},
                        {false, 1.0f / 512, 0, 0, 0},
                        {false, 5.0f / 512, 0, 0, 2.0f / 256},
                        {false, 3.0f / 512, 0, 0, 2.0f / 256},
                        {false, 1.0f / 512, 1, 0, 1.0f / 256},
                        {false, -1.0f / 512, 1, 0, 1.0f / 256},
                        {false, -INFINITY, 256, 0, 0},
                        {false, INFINITY, -256, 0, 3},
                        {false, -INFINITY, 0x1fff, 256, 1},
                        {false, INFINITY, -0x2000, 256, 3},
                        {false, NAN, 256, 0, 1},
                        {false, -1.0e30f, 256, 0, 0},
                        {false, 1.0e30f, -256, 0, 3},
                        {false, 0, 0, 384, 1.5f}};
  wf_->set_exec(1);
  for (const Case &test : cases)
    for (MipFilter mip_filter : {MipFilter::Point, MipFilter::Linear}) {
      SCOPED_TRACE(testing::Message() << test.zero << ',' << test.lod << ',' << test.bias << ','
                                      << test.minimum << ',' << static_cast<uint32_t>(mip_filter));
      bind_sampler({.mip = mip_filter, .min_lod_fixed8 = test.minimum, .bias_fixed8 = test.bias});
      cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(0.5f));
      cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
      cu_->write_vgpr(vb_ + 2, 0, std::bit_cast<uint32_t>(test.lod));
      ASSERT_TRUE(sample(test.zero ? SampleTestOperation::Zero : SampleTestOperation::Lod));
      const float expected =
          mip_filter == MipFilter::Linear ? test.expected : std::floor(test.expected + 0.5f);
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(expected));
    }
  // Independent GFX11/12 captures distinguish integer-LOD mip blending.
  for (uint32_t level = 0; level < 4; ++level)
    for (uint32_t y = 0; y < (16u >> level); ++y)
      for (uint32_t x = 0; x < (16u >> level); ++x)
        memory_.write32(kBase + offsets[level] + y * (gfx12 ? 128 : 256) + x * 4, 0x80000000u);
  for (bool linear : {false, true})
    for (MipFilter mip_filter : {MipFilter::Point, MipFilter::Linear})
      for (float lod : {0.0f, 0.25f, 0.5f, 1.0f, 1.5f, 2.75f, 3.0f}) {
        bind_sampler({.linear_xy = linear, .mip = mip_filter});
        cu_->write_vgpr(vb_, 0, 0u);
        cu_->write_vgpr(vb_ + 1, 0, 0u);
        cu_->write_vgpr(vb_ + 2, 0, std::bit_cast<uint32_t>(lod));
        ASSERT_TRUE(sample(SampleTestOperation::Lod));
        const bool blends = mip_filter == MipFilter::Linear && lod != std::floor(lod);
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), blends ? 0u : 0x80000000u);
      }
  struct BlendCase {
    uint64_t exec;
    uint32_t fractional_lane;
    float lod;
    uint64_t gfx12_positive;
    uint64_t other_positive;
  };
  // Captured quad contexts: a scalar witness cannot distinguish these rules.
  const BlendCase blends[] = {{15, 8, 0.25f, 0, 0},
                              {15, 3, 0.25f, 15, 8},
                              {7, 3, 0.25f, 0, 0},
                              {255, 4, 0.25f, 240, 16},
                              {15, 3, 1.0f + 1.0f / 512, 0, 0},
                              {15, 3, 3.5f, 0, 0},
                              {2, 8, 0.25f, 0, 0}};
  for (bool linear : {false, true})
    for (const BlendCase &test : blends) {
      SCOPED_TRACE(testing::Message()
                   << test.exec << ',' << test.fractional_lane << ',' << test.lod);
      wf_->set_exec(test.exec);
      bind_sampler({.linear_xy = linear, .mip = MipFilter::Linear});
      for (uint32_t lane = 0; lane < 8; ++lane) {
        cu_->write_vgpr(vb_, lane, 0u);
        cu_->write_vgpr(vb_ + 1, lane, 0u);
        cu_->write_vgpr(vb_ + 2, lane,
                        std::bit_cast<uint32_t>(lane == test.fractional_lane ? test.lod : 1.0f));
        cu_->write_vgpr(vb_ + 8, lane, 0xA5A5A5A5u);
      }
      ASSERT_TRUE(sample(SampleTestOperation::Lod));
      for (uint32_t lane = 0; lane < 8; ++lane) {
        const uint64_t bit = uint64_t{1} << lane;
        const uint64_t positive = gfx12 ? test.gfx12_positive : test.other_positive;
        const uint32_t expected = !(test.exec & bit) ? 0xA5A5A5A5u
                                  : positive & bit   ? 0u
                                                     : 0x80000000u;
        EXPECT_EQ(cu_->read_vgpr(vb_ + 8, lane), expected);
      }
    }
}

TEST_P(ImageSamplerControlTest, NullGatherClearsEveryPackedAndUnpackedDestination) {
  if (arch() == ROCJITSU_CODE_ARCH_CDNA2)
    GTEST_SKIP() << "The public CDNA2 ISA and installed SDK omit gather forms";
  wf_->set_exec(5);
  for (bool d16 : {false, true})
    for (uint32_t invalid : {0u, 1u, 2u}) {
      SCOPED_TRACE(testing::Message() << d16 << ',' << invalid);
      Image image;
      image.width = 2;
      bind_sample_image(image, false);
      memory_.write32(kBase, 0x11);
      memory_.write32(kBase + 4, 0x33);
      if (invalid == 0)
        for (uint32_t word = 0; word < 8; ++word)
          cu_->write_sgpr(sb_ + word, 0);
      if (invalid == 1) {
        cu_->write_sgpr(sb_ + 1, 0);
      }
      bind_sampler();
      for (uint32_t lane : {0u, 1u, 2u}) {
        cu_->write_vgpr(vb_, lane, std::bit_cast<uint32_t>(0.5f));
        cu_->write_vgpr(vb_ + 1, lane, std::bit_cast<uint32_t>(0.5f));
        for (uint32_t reg = 0; reg < 4; ++reg)
          cu_->write_vgpr(vb_ + 8 + reg, lane, 0xdeadbeef);
      }
      ASSERT_TRUE(sample(SampleTestOperation::Gather, d16, invalid == 2 ? 104 : 8));
      for (uint32_t lane : {0u, 1u, 2u})
        for (uint32_t reg = 0; reg < 4; ++reg)
          EXPECT_EQ(cu_->read_vgpr(vb_ + 8 + reg, lane),
                    lane != 1 && reg < (d16 ? 2u : 4u) ? 0u : 0xdeadbeefu);
    }
}

TEST_P(ImageSamplerControlTest, IntegerFilteringIsExplicitlyUnsupportedButPointSamplingWorks) {
  Image image;
  image.width = 2;
  bind_sample_image(image, false);
  memory_.write32(kBase, 1);
  memory_.write32(kBase + 4, 3);
  wf_->set_exec(1);
  cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(0.5f));
  cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
  for (Sampler sampler :
       {Sampler{}, Sampler{.linear_xy = true}, Sampler{.mip = MipFilter::Linear}}) {
    bind_sampler(sampler);
    wf_->clear_instruction_execution_error();
    if (!sampler.linear_xy && sampler.mip == MipFilter::None) {
      ASSERT_TRUE(sample(SampleTestOperation::Zero));
      EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), 3u);
    } else {
      EXPECT_FALSE(sample(SampleTestOperation::Zero));
      EXPECT_EQ(wf_->instruction_execution_error(),
                amdgpu::InstructionExecutionError::UnimplementedInstruction);
    }
  }
}

TEST_P(ImageSamplerControlTest, ClampedLinearTapsPreserveNegativeZero) {
  Image image;
  image.width = 2;
  bind_sample_image(image);
  memory_.write32(kBase, 0x80000000u);
  memory_.write32(kBase + 4, 0x80000000u);
  wf_->set_exec(1);
  bind_sampler({.linear_xy = true, .normalized = false});
  for (float x : {0.0f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f}) {
    cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(x));
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), x > 0.5f && x < 1.5f ? 0u : 0x80000000u);
  }
  struct Case {
    uint32_t width;
    Clamp wrap;
    float x;
    uint32_t expected;
  };
  // Captured normalized mirror/repeat edge and distinct-texel controls.
  const Case cases[] = {{2, Clamp::Mirror, 0, 0x80000000u},
                        {4, Clamp::Mirror, 0.05f, 0x80000000u},
                        {4, Clamp::Mirror, 1, 0x80000000u},
                        {1, Clamp::Repeat, 0, 0x80000000u},
                        {1, Clamp::Repeat, -0.3f, 0x80000000u},
                        {1, Clamp::Mirror, 1.3f, 0x80000000u},
                        {2, Clamp::Repeat, 0, 0},
                        {2, Clamp::Mirror, 0.5f, 0},
                        {2, Clamp::Border, 0, 0}};
  for (const Case &test : cases) {
    SCOPED_TRACE(testing::Message()
                 << test.width << ',' << static_cast<uint32_t>(test.wrap) << ',' << test.x);
    image.width = test.width;
    bind_sample_image(image);
    for (uint32_t x = 0; x < image.width; ++x)
      memory_.write32(kBase + x * 4, 0x80000000u);
    bind_sampler({.clamp_x = test.wrap, .linear_xy = true});
    cu_->write_vgpr(vb_, 0, std::bit_cast<uint32_t>(test.x));
    cu_->write_vgpr(vb_ + 1, 0, std::bit_cast<uint32_t>(0.5f));
    ASSERT_TRUE(sample(SampleTestOperation::Zero));
    EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), test.expected);
  }
}

TEST_P(ImageSamplerControlTest, Gfx10CompressedSeamlessCubeMatchesUncompressedCornersAndEdges) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA1 && arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP() << "GFX10 BC cube sampler";
  Image image;
  image.type = amdgpu::kImageCube;
  image.depth = 6;
  image.width = image.height = 4;
  dimension_override_ = amdgpu::kImageDimCube;
  bind_sampler({.linear_xy = true});
  wf_->set_exec(1);
  std::array<std::array<float, 3>, 4> coords{
      {{1.f, 1.f, 0}, {1.03125f, 1.03125f, 3}, {2.f, 1.5f, 4}, {1.375f, 1.625f, 2}}};
  std::array<uint32_t, 4> expected{};
  for (bool bc : {false, true}) {
    image.rdna_format = bc ? 169 : 22; // BC1_UNORM and R32_FLOAT
    bind_image(image);
    for (unsigned face = 0; face < 6; ++face) {
      uint32_t indices = 0;
      for (unsigned texel = 0; texel < 16; ++texel) {
        const bool green = (texel + 3 * face) % 5 < 2;
        if (bc)
          indices |= uint32_t(green) << (2 * texel);
        else
          memory_.write32(kBase + (face * 4 + texel / 4) * 256 + (texel % 4) * 4,
                          std::bit_cast<uint32_t>(green ? 0.f : 1.f));
      }
      if (bc) {
        memory_.write32(kBase + face * 256, 0x07e0f800);
        memory_.write32(kBase + face * 256 + 4, indices);
      }
    }
    cu_->l1_vector().invalidate_all();
    l2_.invalidate_all();
    for (unsigned i = 0; i < coords.size(); ++i) {
      write_coordinates(0, std::bit_cast<std::array<uint32_t, 3>>(coords[i]));
      ASSERT_TRUE(sample(SampleTestOperation::Zero));
      const uint32_t actual = cu_->read_vgpr(vb_ + 8, 0);
      if (bc)
        EXPECT_EQ(actual, expected[i]) << i;
      else
        expected[i] = actual;
    }
  }
}

TEST_P(ImageSamplerControlTest, Gfx10CubeSamplerAndRawAccessSelectFacesWithinArrayView) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA1 && arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP() << "GFX10 cube coordinates";
  Image image;
  image.width = image.height = 1;
  image.type = amdgpu::kImageCube;
  image.depth = 18;
  bind_sample_image(image);
  cu_->write_sgpr(sb_ + 4, cu_->read_sgpr(sb_ + 4) | (6u << 16));
  dimension_override_ = amdgpu::kImageDimCube;
  bind_sampler();
  wf_->set_exec(1);
  // Each linear 1x1 R32 face occupies an aligned row (256 bytes). Skip
  // the first six faces, then select face +Y of the second visible cube.
  memory_.write32(kBase + 14 * 256, std::bit_cast<uint32_t>(14.0f));
  write_coordinates(0, {0, 0, 8});
  ASSERT_TRUE(decoded_issue(true, 3));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(14.0f));
  // Sampler coordinates have the +1 cube bias, and the cube index starts at bit 3.
  write_coordinates(0, {std::bit_cast<uint32_t>(1.5f), std::bit_cast<uint32_t>(1.5f),
                        std::bit_cast<uint32_t>(10.0f)});
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(14.0f));
}

TEST_P(ImageSamplerControlTest, Gfx10PredefinedWhiteBorderReturnsOne) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA1 && arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP() << "GFX10 sampler border colors";
  Image image;
  bind_sample_image(image);
  bind_sampler({.clamp_x = Clamp::Border, .normalized = false});
  cu_->write_sgpr(sb_ + 11, 2u << 30);
  wf_->set_exec(1);
  write_coordinates(0, {std::bit_cast<uint32_t>(-2.0f), 0, 0});
  ASSERT_TRUE(sample(SampleTestOperation::Zero));
  EXPECT_EQ(cu_->read_vgpr(vb_ + 8, 0), std::bit_cast<uint32_t>(1.0f));
}

TEST_P(ImageSamplerControlTest, Gfx10RejectsWholeImageResultsThatCrossTheWaveAllocation) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA1 && arch() != ROCJITSU_CODE_ARCH_RDNA2)
    GTEST_SKIP() << "GFX10 VGPR allocation rule";
  wf_->halt();
  wf_ = cu_->dispatch_wf(0, 0, 106, 16);
  ASSERT_NE(wf_, nullptr);
  sb_ = wf_->sgpr_alloc().base;
  vb_ = wf_->vgpr_alloc().base;
  ASSERT_EQ(wf_->vgpr_alloc().count, 16u);
  ASSERT_EQ(cu_->vgpr_allocation_block_size(), 32u);
  bind_sample_image(Image{});
  bind_sampler();
  wf_->set_exec(1);
  const uint32_t last = wf_->num_vgprs() - 1;
  auto encoding = fields<rdna1::MimgBuilderFields>(2);
  encoding.vdata = last;
  encoding.ssamp = 2;
  encoding.dmask = 3;
  memory_.write32(kBase, 12345u);
  auto decoder = Decoder::create(arch());
  const std::array<std::pair<uint16_t, bool>, 3> cases{{{rdna1::kImageGetResinfoMimg, false},
                                                        {rdna1::kImageLoadMimg, false},
                                                        {rdna1::kImageLoadMimg, true}}};
  for (const auto &[op, tfe] : cases) {
    if (tfe && arch() != ROCJITSU_CODE_ARCH_RDNA1)
      continue;
    encoding.tfe = tfe;
    encoding.dmask = tfe ? 1 : 3;
    cu_->write_vgpr(vb_ + last, 0, 0xdeadbeef);
    auto decoded = decoder->decode_window(rdna1::build_mimg(op, encoding));
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wf_).succeeded());
    if (op == rdna1::kImageLoadMimg) {
      amdgpu::GlobalMemPipeline pipeline(&cu_->l1_vector(), &l2_);
      ASSERT_EQ(pipeline.issue(inst.release(), *wf_), amdgpu::VmAccessOutcome::Complete);
    }
    EXPECT_EQ(cu_->read_vgpr(vb_ + last, 0), 0xdeadbeefu);
  }
}

TEST_P(ImageSamplerControlTest, DerivativeMetadataUsesTwoCubeComponents) {
  if (arch_is_cdna_4_or_lower(arch()) || arch() == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "MIMG DIM encoding on GFX10/GFX11";
  for (uint32_t dim : {1u, 2u, 3u}) {
    dimension_override_ = dim;
    std::unique_ptr<Instruction> instruction = sample_instruction(SampleTestOperation::Gradients);
    ASSERT_TRUE(instruction);
    amdgpu::RegisterModifiers modifiers{};
    instruction->amdgpu_register_modifiers(modifiers);
    const uint32_t prefix = dim == 2 ? 6 : 4;
    const uint32_t coordinates = dim == 1 ? 2 : 3;
    EXPECT_EQ(modifiers.image_prefix_words, prefix);
    EXPECT_EQ(modifiers.image_coordinate_count, coordinates);
    InstDefUse uses(*instruction, 32);
    for (uint32_t reg = 0; reg < prefix + coordinates; ++reg)
      EXPECT_TRUE(uses.uses.contains({RegClass::VGPR, static_cast<uint16_t>(reg), 1}));
    EXPECT_FALSE(
        uses.uses.contains({RegClass::VGPR, static_cast<uint16_t>(prefix + coordinates), 1}));
  }
}

TEST_P(ImageSamplerControlTest, VimageMipTracksItsFifthIndependentSelector) {
  if (arch() != ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "VIMAGE is a GFX12 encoding";
  rdna4::VimageBuilderFields encoding{};
  encoding.dim = 7;
  encoding.dmask = 1;
  encoding.vdata = 8;
  encoding.rsrc = 0;
  encoding.vaddr0 = 1;
  encoding.vaddr1 = 9;
  encoding.vaddr2 = 16;
  encoding.vaddr3 = 23;
  encoding.vaddr4 = 30;
  const auto words = rdna4::build_vimage(rdna4::kImageLoadMipVimage, encoding);
  std::unique_ptr<Decoder> decoder = Decoder::create(arch());
  DecodeResult decoded = decoder->decode(words.data());
  ASSERT_TRUE(decoded.succeeded());
  amdgpu::RegisterModifiers modifiers{};
  decoded.value()->amdgpu_register_modifiers(modifiers);
  EXPECT_EQ(modifiers.image_coordinate_count, 5);
  for (uint32_t i = 0; i < 5; ++i)
    EXPECT_EQ(modifiers.image_coordinates[i], (std::array<uint32_t, 5>{1, 9, 16, 23, 30})[i]);
}

INSTANTIATE_TEST_SUITE_P(Architectures, ImageSamplerControlTest,
                         testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                         ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                         ROCJITSU_CODE_ARCH_RDNA4));

} // namespace
