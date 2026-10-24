// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mxfp4_simd.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace {

using namespace rocjitsu;
struct ForceScalarGuard {
  explicit ForceScalarGuard(bool force_scalar) : old_force_scalar(util::force_scalar()) {
    util::set_force_scalar_for_testing(force_scalar);
  }
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(old_force_scalar); }
  bool old_force_scalar;
};

struct ConversionFixture {
  amdgpu::GpuMemory memory{"conversion_mem"};
  amdgpu::L2Cache l2{"conversion_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vbase = 0;
  ConversionFixture() {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA4;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("conversion_cu", config, &memory, &l2);
    wf = cu->dispatch_wf(0, 0, 106, 256, 64);
    if (wf)
      vbase = wf->vgpr_alloc().base;
  }
  std::vector<uint32_t> snapshot(uint32_t first, uint32_t count) const {
    std::vector<uint32_t> values;
    for (uint32_t reg = first; reg < first + count; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        values.push_back(cu->read_vgpr(vbase + reg, lane));
    return values;
  }
};

TEST(Mxfp4Simd, EveryBf16AndScaleMatchesScalar) {
#if __has_include(<experimental/simd>)
  using U = util::native<uint32_t>;
  if (U::size() <= 1)
    GTEST_SKIP() << "native SIMD unavailable";
  for (uint32_t scale = 0; scale < 255; ++scale)
    for (uint32_t base = 0; base < 65536; base += U::size()) {
      const U input([&](auto lane) { return base + uint32_t(lane); });
      const U actual = amdgpu::bf16_to_mxfp4_simd(input, U(scale), ~uint64_t{0});
      for (uint32_t lane = 0; lane < U::size(); ++lane) {
        const auto expected = util::bf16_to_fp4_e2m1_scaled_rne(base + lane, scale);
        ASSERT_EQ(uint32_t(actual[lane]), expected)
            << "input=" << base + lane << " scale=" << scale;
      }
    }
#else
  GTEST_SKIP() << "native SIMD unavailable";
#endif
}

TEST(Mxfp4Simd, InactiveExceptionalInputsDoNotRaiseHostExceptions) {
#if __has_include(<experimental/simd>)
  using U = util::native<uint32_t>;
  if (U::size() <= 1)
    GTEST_SKIP() << "native SIMD unavailable";
  if (U::size() < 2)
    GTEST_SKIP() << "test needs inactive SIMD lanes";
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  struct RestoreEnvironment {
    std::fenv_t *saved;
    ~RestoreEnvironment() { std::fesetenv(saved); }
  } restore{&saved};
  const U input(
      [](auto lane) { return lane == 0 ? 0x3f80u : (lane % 2 == 0 ? 0x0001u : 0x7f81u); });
  const U scales([](auto lane) { return lane == 0 ? 127u : 254u; });
  std::feclearexcept(FE_ALL_EXCEPT);
  const U result = amdgpu::bf16_to_mxfp4_simd(input, scales, 1);
  EXPECT_EQ(uint32_t(result[0]), 2u);
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
#else
  GTEST_SKIP() << "native SIMD unavailable";
#endif
}

RJ_NOINLINE uint8_t scalar_conversion(uint16_t input, uint8_t scale) {
  return util::bf16_to_fp4_e2m1_scaled_rne(input, scale);
}

TEST(Mxfp4Simd, HostRoundingAndDenormalModesMatchScalar) {
#if __has_include(<experimental/simd>) && defined(__SSE2__)
  using U = util::native<uint32_t>;
  if (U::size() <= 1)
    GTEST_SKIP() << "native SIMD unavailable";
  struct HostMxcsrGuard {
    uint32_t saved = _mm_getcsr();
    ~HostMxcsrGuard() { _mm_setcsr(saved); }
  } guard;
  constexpr std::array<uint16_t, 16> inputs = {0x0000, 0x8000, 0x0001, 0x8040, 0x007f, 0x0080,
                                               0x7f81, 0xff80, 0x3e80, 0xbf40, 0x3fa0, 0xbfe0,
                                               0x4020, 0xc060, 0x40a0, 0x7f7f};
  for (uint32_t mode = 0; mode < 16; ++mode) {
    _mm_setcsr((guard.saved & ~0xe040u) | ((mode & 3u) << 13) | ((mode & 4u) ? 0x8000u : 0u) |
               ((mode & 8u) ? 0x40u : 0u));
    for (uint32_t scale : {0u, 1u, 2u, 126u, 127u, 128u, 253u, 254u})
      for (uint32_t base = 0; base < inputs.size(); base += U::size()) {
        const U input([&](auto lane) { return uint32_t(inputs[(base + lane) % inputs.size()]); });
        const U actual = amdgpu::bf16_to_mxfp4_simd(input, U(scale), ~uint64_t{0});
        for (uint32_t lane = 0; lane < U::size(); ++lane)
          ASSERT_EQ(uint32_t(actual[lane]), scalar_conversion(input[lane], scale))
              << "mode=" << mode << " scale=" << scale << " input=" << input[lane];
      }
  }
#else
  GTEST_SKIP() << "requires native SIMD and x86 host modes";
#endif
}

using Access = std::tuple<char, uint32_t, uint32_t, uint8_t, uint32_t>;

class ConversionObserver final : public ExecutionPlugin {
public:
  ConversionObserver(uint32_t base, const amdgpu::ComputeUnitCore &cu)
      : ExecutionPlugin("conversion"), base_(base), cu_(cu) {}
  std::set<Access> accesses;
  std::set<uint32_t> scalar_reads;

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t reg, uint64_t lanes,
                             uint8_t bytes) override {
    record('r', wf, reg, lanes, bytes);
  }
  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t reg, uint64_t lanes,
                              uint8_t bytes) override {
    record('w', wf, reg, lanes, bytes);
  }
  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t reg) override {
    scalar_reads.insert(reg - wf->sgpr_alloc().base);
  }

private:
  void record(char kind, const amdgpu::Wavefront * /*wf*/, uint32_t reg, uint64_t lanes,
              uint8_t bytes) {
    while (lanes) {
      const uint32_t lane = std::countr_zero(lanes);
      lanes &= lanes - 1;
      accesses.emplace(kind, reg - base_, lane, bytes, cu_.read_vgpr_storage(reg, lane));
    }
  }
  uint32_t base_;
  const amdgpu::ComputeUnitCore &cu_;
};

struct ConversionResult {
  std::vector<uint32_t> registers;
  std::set<Access> accesses;
  std::set<uint32_t> scalar_reads;
  uint64_t exec = 0;
};

ConversionResult execute_conversion(bool force_scalar, uint16_t data_selector,
                                    uint16_t scale_selector, uint32_t destination, uint32_t byte,
                                    uint64_t exec, uint64_t write_mask = ~uint64_t{0},
                                    bool logical_zero = false, bool invalid_scale = false) {
  ForceScalarGuard guard(force_scalar);
  ConversionFixture fx;
  EXPECT_NE(fx.wf, nullptr);
  if (!fx.wf)
    return {};
  fx.wf->set_exec(exec);
  fx.wf->set_vgpr_write_mask(write_mask);
  constexpr std::array<uint16_t, 16> input = {0x0000, 0x8000, 0x0001, 0x807f, 0x0080, 0x7f81,
                                              0xffc0, 0xff80, 0x3e80, 0xbf40, 0x3fa0, 0xbfe0,
                                              0x4020, 0xc060, 0x40a0, 0x7f7f};
  constexpr std::array<uint8_t, 8> scale = {0, 1, 126, 127, 128, 253, 254, 255};
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!logical_zero)
      fx.cu->write_vgpr(fx.vbase, lane,
                        uint32_t(input[lane % input.size()]) |
                            (uint32_t(input[(lane * 7 + 3) % input.size()]) << 16));
    fx.cu->write_vgpr(fx.vbase + 1, lane,
                      (uint32_t(invalid_scale ? 255 : scale[lane % scale.size()]) << 23) |
                          ((lane & 1u) << 31) | (lane * 1237u));
    fx.cu->write_vgpr(fx.vbase + 2, lane, 0xa1b2c3d4u ^ lane);
  }
  fx.cu->write_sgpr(fx.wf->sgpr_alloc().base, 0x3fa0bf40u);
  fx.cu->write_sgpr(fx.wf->sgpr_alloc().base + 1, invalid_scale ? 0x7f800000u : 0x3f800000u);
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<ConversionObserver>(fx.vbase, *fx.cu);
  auto *observations = observer.get();
  group->add(std::move(observer));
  fx.cu->set_plugin_group(group);
  auto words = cdna4::build_vop3(cdna4::kVCvtScalef32PkFp4Bf16Vop3,
                                 {.vdst = static_cast<uint8_t>(destination),
                                  .op_sel = static_cast<uint8_t>(byte << 2),
                                  .src0 = data_selector,
                                  .src1 = scale_selector});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  EXPECT_NE(inst, nullptr);
  if (!inst)
    return {};
  EXPECT_TRUE(fx.cu->execute_instruction(inst.get(), *fx.wf).succeeded());
  // Inspecting results must not add register reads to the instruction's trace.
  fx.cu->set_plugin_group(nullptr);
  return {fx.snapshot(0, 3), observations->accesses, observations->scalar_reads, fx.wf->exec()};
}

template <typename... Args> void compare_conversion(Args... args) {
  const auto scalar = execute_conversion(true, args...);
  const auto simd = execute_conversion(false, args...);
  EXPECT_EQ(simd.registers, scalar.registers);
  EXPECT_EQ(simd.accesses, scalar.accesses);
  EXPECT_EQ(simd.scalar_reads, scalar.scalar_reads);
  EXPECT_EQ(simd.exec, scalar.exec);
}

TEST(Mxfp4Simd, AliasesMasksAndObserverValuesMatchScalar) {
  if (!util::has_stdx_simd || util::native<uint32_t>::size() <= 1)
    GTEST_SKIP() << "native SIMD unavailable";
  for (uint32_t destination : {0u, 1u, 2u})
    for (uint32_t byte = 0; byte < 4; ++byte)
      for (uint64_t exec : {~uint64_t{0}, uint64_t{0xa55aa55aa55aa55a}, uint64_t{0}}) {
        SCOPED_TRACE(testing::Message()
                     << "dst=" << destination << " byte=" << byte << " exec=" << exec);
        compare_conversion(256, 257, destination, byte, exec, uint64_t{0xff00ff00ff00ff00});
      }
  compare_conversion(256, 257, 0, 3, ~uint64_t{0}, ~uint64_t{0}, true);
}

TEST(Mxfp4Simd, ScalarImmediateAndInvalidScaleSourcesMatchScalar) {
  if (!util::has_stdx_simd || util::native<uint32_t>::size() <= 1)
    GTEST_SKIP() << "native SIMD unavailable";
  // This opcode accepts inline constants, but neither source permits literals.
  constexpr std::array<std::pair<uint16_t, uint16_t>, 7> selectors = {
      {{0, 257}, {256, 1}, {0, 1}, {242, 1}, {128, 1}, {193, 1}, {256, 242}}};
  for (auto [data, scale] : selectors)
    for (uint32_t byte = 0; byte < 4; ++byte) {
      SCOPED_TRACE(testing::Message() << "data=" << data << " scale=" << scale << " byte=" << byte);
      compare_conversion(data, scale, byte % 3, byte, uint64_t{0xa55aa55aa55aa55a});
    }
  for (uint16_t data : {uint16_t{0}, uint16_t{256}}) {
    const auto result =
        execute_conversion(false, data, 257, 2, 0, ~uint64_t{0}, ~uint64_t{0}, false, true);
    EXPECT_TRUE(result.scalar_reads.empty());
    for (auto [kind, reg, lane, bytes, value] : result.accesses) {
      EXPECT_NE(reg, 0u);
      EXPECT_EQ(bytes, 15u);
    }
    compare_conversion(data, 257, 2, 0, ~uint64_t{0}, ~uint64_t{0}, false, true);
  }
}

} // namespace
