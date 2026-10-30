// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A DS instruction may complete its LDS access inside execute_impl() instead of
// being routed to the local memory pipeline. Both paths run the same access
// code, so they must leave identical registers, LDS and wave state, and any
// observer of the routed order or an attached debugger must keep the routed path.

#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

constexpr uint64_t kCodeBase = 0x10000;

/// @brief The target a DsCu emulates: CDNA4 (gfx950) has SRAM ECC, which makes
/// D16 loads zero the other half; RDNA4 (gfx1201) does not, and runs wave32.
struct Target {
  rj_code_arch_t arch;
  uint32_t s_endpgm;
  uint32_t wave;
};
constexpr Target kCdna4{ROCJITSU_CODE_ARCH_CDNA4, 0xBF810000u, 64};
constexpr uint32_t kVgprs = 256;
constexpr uint32_t kLdsBytes = 64 * 1024;

uint32_t seed_word(uint32_t index) { return index * 0x9E3779B9u ^ 0x5A5A0F0Fu; }

/// @brief Lane addresses: distinct and permuted, all lanes equal, or partly
/// past the allocation so both paths take their bounds handling.
enum class Addresses { Permuted, Broadcast, PastEnd };

/// @brief One CU running a single DS instruction followed by s_endpgm.
class DsCu {
public:
  explicit DsCu(bool lds_inline, Target target = kCdna4)
      : memory_("ds_inline_mem"), l2_("ds_inline_l2"), target_(target) {
    ComputeUnitCore::Config config{};
    config.arch = target.arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = kVgprs;
    config.lds_size_kb = kLdsBytes / 1024;
    l2_.set_backing_memory(&memory_);
    cu_ = ComputeUnitCore::create("ds_inline_cu", config, &memory_, &l2_);
    cu_->set_lds_inline_for_test(lds_inline);
  }

  /// Execute @p ds once from a seeded wave and LDS.
  void run(const std::array<uint32_t, 2> &ds, uint64_t exec, Addresses addresses) {
    memory_.write32(kCodeBase, ds[0]);
    memory_.write32(kCodeBase + 4, ds[1]);
    memory_.write32(kCodeBase + 8, target_.s_endpgm);
    cu_->begin_workgroup(1, 0, 1);
    wf_ = cu_->dispatch_wf(0, kCodeBase, 106, kVgprs, target_.wave);
    ASSERT_NE(wf_, nullptr);
    wf_->set_lds_base(cu_->allocate_lds(kLdsBytes));
    wf_->set_lds_size(kLdsBytes);
    auto &lds = wf_->lds();
    lds.materialize_range(0, lds.size_bytes());
    for (uint32_t addr = 0; addr + 4 <= lds.size_bytes(); addr += 4)
      lds.write32(addr, seed_word(addr));
    const uint32_t v0 = wf_->vgpr_alloc().base;
    for (uint32_t reg = 1; reg < kVgprs; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        cu_->write_vgpr(v0 + reg, lane, seed_word(reg * 64 + lane));
    for (uint32_t lane = 0; lane < 64; ++lane) {
      uint32_t address = ((lane * 7) % 64) * 16 + 64;
      if (addresses == Addresses::Broadcast)
        address = 256;
      else if (addresses == Addresses::PastEnd && lane % 4 == 3)
        address = kLdsBytes - 8 + lane;
      cu_->write_vgpr(v0, lane, address);
    }
    wf_->set_m0(0x100);
    wf_->set_exec(exec);
    cu_->step();
  }

  /// Registers, LDS and wave state that either path could have changed.
  std::vector<uint32_t> snapshot() const {
    std::vector<uint32_t> out;
    const uint32_t v0 = wf_->vgpr_alloc().base;
    for (uint32_t reg = 0; reg < kVgprs; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        out.push_back(cu_->read_vgpr(v0 + reg, lane));
    const auto &lds = wf_->lds();
    for (uint32_t addr = 0; addr + 4 <= lds.size_bytes(); addr += 4)
      out.push_back(lds.read32(addr));
    const auto &counters = wf_->wait_counters();
    out.push_back(counters.vmcnt);
    out.push_back(counters.lgkmcnt);
    out.push_back(static_cast<uint32_t>(wf_->pc - kCodeBase));
    out.push_back(static_cast<uint32_t>(wf_->state()));
    return out;
  }

  ComputeUnitCore &cu() { return *cu_; }

private:
  GpuMemory memory_;
  L2Cache l2_;
  Target target_;
  std::unique_ptr<ComputeUnitCore> cu_;
  Wavefront *wf_ = nullptr;
};

/// @brief Run every decodable DS opcode on @p target both ways and require
/// identical results. Returns {compared, inline completions}.
template <typename Build>
std::pair<unsigned, unsigned> compare_every_ds_opcode(Target target, Build build) {
  auto decoder = Decoder::create(target.arch);
  unsigned compared = 0, inline_completions = 0;
  // Ordinary operands, then data and destination registers at the end of the
  // allocation, so multi-register forms reach past it.
  struct Operands {
    uint8_t data0, data1, vdst;
  };
  for (uint16_t op = 0; op < 256; ++op)
    for (const Operands operands : {Operands{1, 2, 8}, Operands{255, 255, 255}}) {
      const std::array<uint32_t, 2> words =
          build(op, operands.data0, operands.data1, operands.vdst);
      DecodeResult decoded = decoder->decode(words.data());
      if (decoded.failed())
        continue;
      const std::string mnemonic(decoded.value()->mnemonic());
      // GWS operations park and wake waves across the device; both paths route them.
      if (mnemonic.starts_with("ds_gws"))
        continue;
      for (const uint64_t exec : {~uint64_t{0}, uint64_t{0xA55AA55AA55AA55Aull}, uint64_t{0}})
        for (const Addresses addresses :
             {Addresses::Permuted, Addresses::Broadcast, Addresses::PastEnd}) {
          SCOPED_TRACE(testing::Message() << mnemonic << " wave" << target.wave << " data0="
                                          << unsigned(operands.data0) << " exec=" << std::hex
                                          << exec << " addresses=" << static_cast<int>(addresses));
          DsCu routed(false, target), inlined(true, target);
          routed.run(words, exec, addresses);
          inlined.run(words, exec, addresses);
          EXPECT_FALSE(routed.cu().lds_inline_completed_for_test());
          inline_completions += inlined.cu().lds_inline_completed_for_test();
          EXPECT_EQ(inlined.snapshot(), routed.snapshot());
          if (testing::Test::HasFailure())
            return {compared, inline_completions};
          ++compared;
        }
    }
  return {compared, inline_completions};
}

TEST(DsInlineCompletion, EveryDsOpcodeMatchesTheRoutedPath) {
  const auto [compared, inline_completions] =
      compare_every_ds_opcode(kCdna4, [](uint16_t op, uint8_t data0, uint8_t data1, uint8_t vdst) {
        return cdna4::build_ds(
            op,
            {.offset0 = 8, .offset1 = 1, .addr = 0, .data0 = data0, .data1 = data1, .vdst = vdst});
      });
  // Ordinary DS loads, stores and atomics must actually take the inline path.
  EXPECT_GT(compared, 600u);
  EXPECT_GT(inline_completions, compared / 2);
}

// RDNA4 has no SRAM ECC, so D16 loads keep the other half of the destination,
// and waves are 32 or 64 lanes; the inline writeback is specialized by both.
TEST(DsInlineCompletion, EveryRdna4DsOpcodeMatchesTheRoutedPathInBothWaveSizes) {
  for (const uint32_t wave : {32u, 64u}) {
    const auto [compared, inline_completions] =
        compare_every_ds_opcode(Target{ROCJITSU_CODE_ARCH_RDNA4, 0xBFB00000u, wave},
                                [](uint16_t op, uint8_t data0, uint8_t data1, uint8_t vdst) {
                                  return rdna4::build_vds(op, {.offset0 = 8,
                                                               .offset1 = 1,
                                                               .addr = 0,
                                                               .data0 = data0,
                                                               .data1 = data1,
                                                               .vdst = vdst});
                                });
    EXPECT_GT(compared, 400u);
    EXPECT_GT(inline_completions, compared / 2);
  }
}

/// @brief Plugin that observes exactly one hook family, so attaching it shows
/// which observer keeps a DS instruction on the routed path.
class OneHookObserver final : public ExecutionPlugin {
public:
  enum class Hook { None, AfterExecute, InstructionRouting, MemoryRouting };
  explicit OneHookObserver(Hook hook) : ExecutionPlugin("ds_inline_gate"), hook_(hook) {}
  bool observes_before_execute_instruction() const override { return false; }
  bool observes_after_execute_instruction() const override { return hook_ == Hook::AfterExecute; }
  bool observes_async_instruction_issued() const override { return false; }
  bool observes_memory_instruction_routing() const override {
    return hook_ == Hook::InstructionRouting;
  }
  bool observes_vgpr_reads() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool observes_sgpr_reads() const override { return false; }
  bool observes_memory_routing() const override { return hook_ == Hook::MemoryRouting; }

private:
  Hook hook_;
};

TEST(DsInlineCompletion, ObserversAndDebuggerKeepTheRoutedPath) {
  const auto words = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 8});
  using Hook = OneHookObserver::Hook;
  for (const Hook hook :
       {Hook::None, Hook::AfterExecute, Hook::InstructionRouting, Hook::MemoryRouting}) {
    for (const bool debugger : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "hook=" << static_cast<int>(hook) << " debugger=" << debugger);
      DsCu fixture(true);
      auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
      group->add(std::make_unique<OneHookObserver>(hook));
      fixture.cu().set_plugin_group(group);
      fixture.cu().set_debug_active(debugger);
      fixture.run(words, ~uint64_t{0}, Addresses::Permuted);
      EXPECT_EQ(fixture.cu().lds_inline_completed_for_test(), hook == Hook::None && !debugger);
    }
  }
}

/// @brief Records VGPR read notifications and nothing else.
class VgprReadRecorder final : public ExecutionPlugin {
public:
  VgprReadRecorder() : ExecutionPlugin("ds_inline_reads") {}
  bool observes_before_execute_instruction() const override { return false; }
  bool observes_after_execute_instruction() const override { return false; }
  bool observes_async_instruction_issued() const override { return false; }
  bool observes_memory_instruction_routing() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool observes_sgpr_reads() const override { return false; }
  void onAmdgpuReadVgprLanes(const Wavefront * /*wf*/, uint32_t reg, uint64_t lanes,
                             uint8_t bytes) override {
    reads.emplace_back(reg, lanes, bytes);
  }
  std::vector<std::tuple<uint32_t, uint64_t, uint8_t>> reads;
};

TEST(DsInlineCompletion, RegisterReadObserversSeeTheRoutedReads) {
  // A read observer keeps plain loads and stores off the request path, which
  // reads raw storage; they complete inline from a VectorMemState instead and
  // report exactly the reads the routed path reports.
  for (const uint16_t op : {cdna4::kDsReadB32Ds, cdna4::kDsReadB64Ds, cdna4::kDsWriteB32Ds,
                            cdna4::kDsWriteB64Ds, cdna4::kDsWriteB16D16HiDs}) {
    SCOPED_TRACE(op);
    const auto words = cdna4::build_ds(op, {.addr = 0, .data0 = 1, .vdst = 8});
    std::vector<std::vector<std::tuple<uint32_t, uint64_t, uint8_t>>> reads;
    for (const bool lds_inline : {false, true}) {
      DsCu fixture(lds_inline);
      auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
      auto recorder = std::make_unique<VgprReadRecorder>();
      auto *recorded = recorder.get();
      group->add(std::move(recorder));
      fixture.cu().set_plugin_group(group);
      fixture.run(words, 0xA55AA55AA55AA55Aull, Addresses::Permuted);
      EXPECT_EQ(fixture.cu().lds_inline_completed_for_test(), lds_inline);
      reads.push_back(recorded->reads);
    }
    EXPECT_FALSE(reads[1].empty());
    EXPECT_EQ(reads[1], reads[0]);
  }
}

} // namespace
