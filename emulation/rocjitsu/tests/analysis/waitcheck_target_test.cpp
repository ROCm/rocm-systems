// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

namespace rocjitsu::waitcheck_detail {
namespace {

using Target = WaitcheckTarget;

std::unique_ptr<Instruction> decode(std::span<const uint32_t> words, rj_code_arch_t arch) {
  auto decoder = Decoder::create(arch);
  if (!decoder) {
    ADD_FAILURE() << "missing model decoder";
    return nullptr;
  }
  util::StringDiagnostic error;
  auto result = decoder->decode_window(words, 0, error.emitter());
  if (result.failed()) {
    ADD_FAILURE() << error.message();
    return nullptr;
  }
  return std::move(result).value();
}

TEST(WaitcheckTarget, SplitWaitCarriesOnlyItsActiveCounter) {
  const std::array<uint32_t, 1> words{0xbfc00003}; // s_wait_loadcnt 3.
  auto inst = decode(words, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(inst, nullptr);
  auto fields_result = Target::explicit_wait_fields(*inst, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(fields_result.succeeded());
  auto fields = std::move(fields_result).value();
  ASSERT_TRUE(fields);
  EXPECT_EQ((*fields)[Target::counter_index(WaitCounterKind::Load)], 3u);
  EXPECT_EQ(std::ranges::count_if(*fields, [](auto field) { return field.has_value(); }), 1);
}

TEST(WaitcheckTarget, NoWaitSentinelIsNotAWait) {
  const std::array<uint32_t, 1> words{0xbfc0003f};
  auto inst = decode(words, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(inst, nullptr);
  auto fields_result = Target::explicit_wait_fields(*inst, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(fields_result.succeeded());
  auto fields = std::move(fields_result).value();
  ASSERT_TRUE(fields);
  EXPECT_TRUE(std::ranges::none_of(*fields, [](auto field) { return field.has_value(); }));
}

TEST(WaitcheckTarget, LegacyWaitSeparatesPackedCounters) {
  const std::array<uint32_t, 1> words{0xbf8c0072}; // vmcnt(2), lgkmcnt(0), expcnt(7).
  auto inst = decode(words, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(inst, nullptr);
  auto fields_result = Target::explicit_wait_fields(*inst, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(fields_result.succeeded());
  auto fields = std::move(fields_result).value();
  ASSERT_TRUE(fields);
  EXPECT_EQ((*fields)[Target::counter_index(WaitCounterKind::Load)], 2u);
  EXPECT_EQ((*fields)[Target::counter_index(WaitCounterKind::Ds)], 0u);
  EXPECT_FALSE((*fields)[Target::counter_index(WaitCounterKind::Exp)]);
}

TEST(WaitcheckTarget, Rdna3DependencyWaitDecodesBothFieldsAndSentinels) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    const std::array<uint32_t, 1> words{0xbf882fe7u};
    auto inst = decode(words, arch);
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->mnemonic(), "s_waitcnt_depctr");
    auto fields_result = Target::explicit_wait_fields(*inst, arch);
    ASSERT_TRUE(fields_result.succeeded());
    auto fields = std::move(fields_result).value();
    ASSERT_TRUE(fields);
    EXPECT_EQ((*fields)[Target::counter_index(WaitCounterKind::VaVdst)], 2u);
    EXPECT_EQ((*fields)[Target::counter_index(WaitCounterKind::VmVsrc)], 1u);

    const std::array<uint32_t, 1> no_wait_words{0xbf88ffffu};
    auto no_wait = decode(no_wait_words, arch);
    ASSERT_NE(no_wait, nullptr);
    fields_result = Target::explicit_wait_fields(*no_wait, arch);
    ASSERT_TRUE(fields_result.succeeded());
    fields = std::move(fields_result).value();
    ASSERT_TRUE(fields);
    EXPECT_TRUE(std::ranges::none_of(*fields, [](auto field) { return field.has_value(); }));
  }
}

TEST(WaitcheckTarget, DependencyWaitExpressionsUseTheTargetMnemonic) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    EXPECT_EQ(wait_expression(WaitCounterKind::VaVdst, 2, arch).value(),
              "s_waitcnt_depctr depctr_va_vdst(2)");
    EXPECT_EQ(wait_expression(WaitCounterKind::VmVsrc, 1, arch).value(),
              "s_waitcnt_depctr depctr_vm_vsrc(1)");
  }
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    EXPECT_EQ(wait_expression(WaitCounterKind::VaVdst, 2, arch).value(),
              "s_wait_alu depctr_va_vdst(2)");
    EXPECT_EQ(wait_expression(WaitCounterKind::VmVsrc, 1, arch).value(),
              "s_wait_alu depctr_vm_vsrc(1)");
  }
}

TEST(WaitcheckTarget, Rdna3LdsDirectLoadsProduceExportEvents) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5}) {
    for (uint32_t word : {0xce000000u, 0xce100000u}) {
      const std::array<uint32_t, 1> words{word};
      auto inst = decode(words, arch);
      ASSERT_NE(inst, nullptr);
      ASSERT_TRUE(inst->mnemonic() == "lds_param_load" || inst->mnemonic() == "lds_direct_load");
      EXPECT_TRUE(inst->is_memory_wait_producer());
      auto events_result = Target::classify_events(*inst, arch);
      ASSERT_TRUE(events_result.succeeded());
      const auto events = std::move(events_result).value();
      ASSERT_EQ(events.size(), 1u);
      EXPECT_EQ(events[0].counter, WaitCounterKind::Exp);
      EXPECT_EQ(events[0].kind, WaitEventKind::LdsDirect);
      EXPECT_EQ(events[0].registers, TrackedRegisterSource::Defs);
      auto fields_result = Target::embedded_wait_fields(*inst, arch);
      ASSERT_TRUE(fields_result.succeeded());
      const auto fields = std::move(fields_result).value();
      ASSERT_TRUE(fields);
      EXPECT_EQ((*fields)[Target::counter_index(WaitCounterKind::VaVdst)], 0u);
      EXPECT_FALSE((*fields)[Target::counter_index(WaitCounterKind::VmVsrc)]);
    }
  }
}

TEST(WaitcheckTarget, LdsDirectEmbeddedWaitFieldsFollowTheirEncodingGeneration) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool legacy = uses_legacy_waitcnt(waitcnt_model(arch).value());
    for (uint32_t opcode : {0xce000000u, 0xce100000u}) {
      for (uint32_t count : {0u, 3u, 15u}) {
        // WAITVDST/WAIT_VA_VDST occupies bits 16-19 on both generations.
        // Bit 23 is reserved on GFX11 and is WAIT_VM_VSRC on GFX12.
        const std::array<uint32_t, 1> words{opcode | (count << 16) | (legacy ? 0u : (1u << 23))};
        auto inst = decode(words, arch);
        ASSERT_NE(inst, nullptr);
        auto fields_result = Target::embedded_wait_fields(*inst, arch);
        ASSERT_TRUE(fields_result.succeeded());
        const auto fields = std::move(fields_result).value();
        if (count == 15) {
          EXPECT_FALSE(fields);
        } else {
          ASSERT_TRUE(fields);
          EXPECT_EQ((*fields)[Target::counter_index(WaitCounterKind::VaVdst)], count);
          EXPECT_FALSE((*fields)[Target::counter_index(WaitCounterKind::VmVsrc)]);
        }
        if (legacy) {
          EXPECT_FALSE(Target::dsdir_wait_vm_vsrc(*inst));
        } else {
          EXPECT_EQ(Target::dsdir_wait_vm_vsrc(*inst), 1u);
        }
      }
    }
  }
}

TEST(WaitcheckTarget, DependencyBoundsExcludeTheNoWaitEncoding) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (size_t i = 0; i < kCounterCount; ++i) {
      const auto counter = static_cast<WaitCounterKind>(i);
      auto sentinel_result = Target::counter_no_wait_value(arch, counter);
      ASSERT_TRUE(sentinel_result.succeeded());
      const auto sentinel = std::move(sentinel_result).value();
      if (sentinel) {
        EXPECT_EQ(Target::maximum_dependency_wait(arch, counter).value() + 1, *sentinel)
            << static_cast<int>(arch) << ' ' << wait_counter_name(counter);
      }
    }
  }
}

TEST(WaitcheckTarget, OrdinaryAluDoesNotEmitEventsOrWaits) {
  const std::array<uint32_t, 1> words{0xbf800000};
  auto inst = decode(words, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(inst, nullptr);
  EXPECT_TRUE(Target::classify_events(*inst, ROCJITSU_CODE_ARCH_RDNA4).value().empty());
  EXPECT_FALSE(Target::explicit_wait_fields(*inst, ROCJITSU_CODE_ARCH_RDNA4).value());
  EXPECT_FALSE(Target::embedded_wait_fields(*inst, ROCJITSU_CODE_ARCH_RDNA4).value());
}

TEST(WaitcheckTarget, Rdna4LoadTracksResultAndSourceLifetimes) {
  const std::array<uint32_t, 3> words{0xee05007c, 0, 2}; // global_load_b32 v0, v[2:3], off.
  auto inst = decode(words, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(inst, nullptr);
  auto events_result = Target::classify_events(*inst, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(events_result.succeeded());
  const auto events = std::move(events_result).value();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(events[0].registers, TrackedRegisterSource::Defs);
  EXPECT_TRUE(events[0].check_uses);
  EXPECT_TRUE(events[0].check_defs);
  EXPECT_EQ(events[1].counter, WaitCounterKind::VmVsrc);
  EXPECT_EQ(events[1].registers, TrackedRegisterSource::VectorUses);
  EXPECT_FALSE(events[1].check_uses);
  EXPECT_TRUE(events[1].check_defs);
}

TEST(WaitcheckTarget, Rdna4StoreSeparatesCompletionAndDataLifetime) {
  const std::array<uint32_t, 3> words{0xee06807c, 0, 2}; // global_store_b32 v[2:3], v0, off.
  auto inst = decode(words, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(inst, nullptr);
  auto events_result = Target::classify_events(*inst, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(events_result.succeeded());
  const auto events = std::move(events_result).value();
  ASSERT_GE(events.size(), 2u);
  auto store = std::ranges::find(events, WaitCounterKind::Store, &ClassifiedEvent::counter);
  ASSERT_NE(store, events.end());
  EXPECT_EQ(store->registers, TrackedRegisterSource::None);
  auto data = std::ranges::find(events, WaitCounterKind::VmVsrc, &ClassifiedEvent::counter);
  ASSERT_NE(data, events.end());
  EXPECT_EQ(data->registers, TrackedRegisterSource::VectorUses);
  EXPECT_FALSE(data->check_uses);
  EXPECT_TRUE(data->check_defs);
}

// Classification is intentionally independent of decoding. These names cover
// async producers whose ordering obligations differ from ordinary register loads.
class NamedInstruction : public Instruction {
public:
  explicit NamedInstruction(std::string_view name) : Instruction(name, nullptr) {}
};

TEST(WaitcheckTarget, AsyncLoadUpdatesBothMemoryAndAsyncCounters) {
  NamedInstruction inst("global_load_async_to_lds_b32");
  auto events_result = Target::classify_events(inst, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(events_result.succeeded());
  const auto events = std::move(events_result).value();
  ASSERT_EQ(events.size(), 3u);
  EXPECT_EQ(events[0].counter, WaitCounterKind::Load);
  EXPECT_EQ(events[1].counter, WaitCounterKind::Async);
  EXPECT_TRUE(events[1].check_memory_order);
  EXPECT_FALSE(events[1].check_program_end);
  EXPECT_EQ(events[2].counter, WaitCounterKind::X);
}

TEST(WaitcheckTarget, TensorOperationsHaveOnlyTensorEvents) {
  for (auto name : {"tensor_load_to_lds", "tensor_store_from_lds"}) {
    NamedInstruction inst(name);
    auto events_result = Target::classify_events(inst, ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_TRUE(events_result.succeeded());
    const auto events = std::move(events_result).value();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].counter, WaitCounterKind::Tensor);
    EXPECT_TRUE(events[0].check_memory_order);
    EXPECT_FALSE(events[0].check_program_end);
  }
}

TEST(WaitcheckTarget, UnsupportedArchitecturesDoNotInheritSplitCounters) {
  NamedInstruction inst("s_nop");
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RV32I, ROCJITSU_CODE_ARCH_RV64I, ROCJITSU_CODE_ARCH_INVALID}) {
    EXPECT_TRUE(waitcnt_model(arch).failed());
    EXPECT_TRUE(Target::maximum_dependency_wait(arch, WaitCounterKind::Load).failed());
    EXPECT_TRUE(wait_expression(WaitCounterKind::Load, 0, arch).failed());
    EXPECT_TRUE(Target::classify_events(inst, arch).failed());
    EXPECT_TRUE(Target::counter_no_wait_value(arch, WaitCounterKind::Load).failed());
    // Each entry point validates independently, even for a non-wait mnemonic.
    EXPECT_TRUE(Target::explicit_wait_fields(inst, arch).failed());
    EXPECT_TRUE(Target::embedded_wait_fields(inst, arch).failed());
  }
  EXPECT_FALSE(Target::counter_no_wait_value(ROCJITSU_CODE_ARCH_RDNA4, WaitCounterKind::X).value());
  EXPECT_FALSE(
      Target::counter_no_wait_value(ROCJITSU_CODE_ARCH_RDNA4, WaitCounterKind::Async).value());
  EXPECT_FALSE(
      Target::counter_no_wait_value(ROCJITSU_CODE_ARCH_RDNA4, WaitCounterKind::Tensor).value());
}

TEST(WaitcheckTarget, AtomicsDistinguishRegisterResultsFromMemoryDestinations) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (bool returns : {false, true}) {
      std::vector<uint32_t> words;
      if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
        words = {returns ? 0xdd098000u : 0xdd088000u, 0x007f0102u};
      else if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
        words = {returns ? 0xdcd64000u : 0xdcd60000u, 0x007c0102u};
      else
        words = {0xee0d4000u, returns ? 0x011c0000u : 0x010c0000u, 1u};
      std::unique_ptr<Instruction> inst = decode(words, arch);
      ASSERT_NE(inst, nullptr);
      auto events_result = Target::classify_events(*inst, arch);
      ASSERT_TRUE(events_result.succeeded());
      const auto events = std::move(events_result).value();
      ASSERT_FALSE(events.empty()) << static_cast<int>(arch);
      EXPECT_EQ(events[0].counter, returns ? WaitCounterKind::Load
                                           : vmem_store_wait_counter(waitcnt_model(arch).value()));
      EXPECT_EQ(events[0].kind,
                returns ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore);
      EXPECT_EQ(events[0].registers,
                returns ? TrackedRegisterSource::Defs : TrackedRegisterSource::None);
      EXPECT_EQ(events[0].check_uses, returns);
    }
  }
}

TEST(WaitcheckTarget, ImageAtomicsUseTheEncodedReturnControl) {
  struct Case {
    rj_code_arch_t arch;
    uint32_t opcode;
    unsigned glc_bit;
  };
  for (const auto &c : {Case{ROCJITSU_CODE_ARCH_CDNA1, 0xf0480100u, 13},
                        Case{ROCJITSU_CODE_ARCH_RDNA1, 0xf0440100u, 13},
                        Case{ROCJITSU_CODE_ARCH_RDNA2, 0xf0440100u, 13},
                        Case{ROCJITSU_CODE_ARCH_RDNA3, 0xf0300100u, 14},
                        Case{ROCJITSU_CODE_ARCH_RDNA3_5, 0xf0300100u, 14},
                        Case{ROCJITSU_CODE_ARCH_RDNA4, 0xd0430000u, 0}}) {
    for (bool returns : {false, true}) {
      SCOPED_TRACE(static_cast<int>(c.arch));
      SCOPED_TRACE(returns);
      const bool split = c.arch == ROCJITSU_CODE_ARCH_RDNA4;
      const std::array<uint32_t, 3> words{c.opcode | (returns && !split ? 1u << c.glc_bit : 0u),
                                          split ? (returns ? 0x00100000u : 0u) : 1u,
                                          split ? 1u : 0u};
      auto inst = decode(words, c.arch);
      ASSERT_NE(inst, nullptr);
      EXPECT_TRUE(inst->is_memory_wait_producer());
      auto events_result = Target::classify_events(*inst, c.arch);
      ASSERT_TRUE(events_result.succeeded());
      const auto events = std::move(events_result).value();
      ASSERT_FALSE(events.empty());
      EXPECT_EQ(events[0].counter, returns
                                       ? WaitCounterKind::Load
                                       : vmem_store_wait_counter(waitcnt_model(c.arch).value()));
      EXPECT_EQ(events[0].check_uses, returns);
      EXPECT_TRUE(std::ranges::none_of(
          events, [](const auto &event) { return event.counter == WaitCounterKind::Exp; }));
    }
  }
}

TEST(WaitcheckTarget, ScalarMemoryFamiliesAccountForCounterOnlyOperations) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (std::string_view name :
         {"s_atomic_add", "s_buffer_atomic_swap", "s_scratch_load_dword", "s_scratch_store_dword",
          "s_dcache_discard", "s_dcache_wb", "s_gl1_inv", "s_memtime", "s_memrealtime",
          "s_get_barrier_state", "s_get_waveid_in_workgroup"}) {
      NamedInstruction inst(name);
      auto events_result = Target::classify_events(inst, arch);
      ASSERT_TRUE(events_result.succeeded());
      const auto events = std::move(events_result).value();
      ASSERT_FALSE(events.empty()) << name;
      EXPECT_EQ(events[0].counter, smem_wait_counter(waitcnt_model(arch).value()));
      EXPECT_EQ(events[0].kind, WaitEventKind::Smem);
      EXPECT_EQ(events[0].registers, TrackedRegisterSource::None);
    }
  }
}

TEST(WaitcheckTarget, DecodedProducerFlagsCoverAllAmdArchitectures) {
  struct Case {
    rj_code_arch_t arch;
    std::vector<uint32_t> words;
    WaitCounterKind counter;
  };
  // Encoded by LLVM MC for gfx908/90a/942/950/1250/1010/1030/1100/1151/1200.
  const Case cases[] = {
      {ROCJITSU_CODE_ARCH_CDNA1,
       {0xd87a0000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_CDNA1,
       {0xd87e0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_CDNA1, {0xd8280000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_CDNA1,
       {0xc0020100u, 0x0u},
       WaitCounterKind::Ds}, // s_load_dword s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_CDNA1,
       {0xdc508000u, 0x27f0000u},
       WaitCounterKind::Load}, // global_load_dword v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_CDNA1,
       {0xdc708000u, 0x7f0200u},
       WaitCounterKind::Load}, // global_store_dword v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_CDNA2,
       {0xd87a0000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_CDNA2,
       {0xd87e0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_CDNA2, {0xd8280000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_CDNA2,
       {0xc0020100u, 0x0u},
       WaitCounterKind::Ds}, // s_load_dword s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_CDNA2,
       {0xdc508000u, 0x27f0000u},
       WaitCounterKind::Load}, // global_load_dword v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_CDNA2,
       {0xdc708000u, 0x7f0200u},
       WaitCounterKind::Load}, // global_store_dword v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_CDNA3,
       {0xd87a0000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_CDNA3,
       {0xd87e0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_CDNA3, {0xd8280000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_CDNA3,
       {0xc0020100u, 0x0u},
       WaitCounterKind::Ds}, // s_load_dword s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_CDNA3,
       {0xdc508000u, 0x27f0000u},
       WaitCounterKind::Load}, // global_load_dword v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_CDNA3,
       {0xdc708000u, 0x7f0200u},
       WaitCounterKind::Load}, // global_store_dword v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_CDNA4,
       {0xd87a0000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_CDNA4,
       {0xd87e0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_CDNA4, {0xd8280000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_CDNA4,
       {0xc0020100u, 0x0u},
       WaitCounterKind::Ds}, // s_load_dword s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_CDNA4,
       {0xdc508000u, 0x27f0000u},
       WaitCounterKind::Load}, // global_load_dword v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_CDNA4,
       {0xdc708000u, 0x7f0200u},
       WaitCounterKind::Load}, // global_store_dword v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_CDNA5,
       {0xd8d40000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_CDNA5,
       {0xdacc0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_CDNA5, {0xd8500000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_CDNA5,
       {0xf4000100u, 0xf8000000u},
       WaitCounterKind::Km}, // s_load_b32 s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_CDNA5,
       {0xee05007cu, 0x2u, 0x0u},
       WaitCounterKind::Load}, // global_load_b32 v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_CDNA5,
       {0xee06807cu, 0x1000000u, 0x0u},
       WaitCounterKind::Store}, // global_store_b32 v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_CDNA5, {0xbe844c80u}, WaitCounterKind::Km}, // s_sendmsg_rtn_b32 s4, 0x80
      {ROCJITSU_CODE_ARCH_RDNA1,
       {0xd8d40000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_RDNA1,
       {0xdacc0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_RDNA1, {0xd8500000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_RDNA1,
       {0xf4000100u, 0xfa000000u},
       WaitCounterKind::Ds}, // s_load_dword s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_RDNA1,
       {0xdc308000u, 0x27d0000u},
       WaitCounterKind::Load}, // global_load_dword v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_RDNA1,
       {0xdc708000u, 0x7d0200u},
       WaitCounterKind::Store}, // global_store_dword v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_RDNA1,
       {0xf4a80100u, 0x0u},
       WaitCounterKind::Ds}, // s_get_waveid_in_workgroup s4
      {ROCJITSU_CODE_ARCH_RDNA2,
       {0xd8d40000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_RDNA2,
       {0xdacc0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_RDNA2, {0xd8500000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_RDNA2,
       {0xf4000100u, 0xfa000000u},
       WaitCounterKind::Ds}, // s_load_dword s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_RDNA2,
       {0xdc308000u, 0x27d0000u},
       WaitCounterKind::Load}, // global_load_dword v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_RDNA2,
       {0xdc708000u, 0x7d0200u},
       WaitCounterKind::Store}, // global_store_dword v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_RDNA3,
       {0xd8d40000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_RDNA3,
       {0xdacc0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_RDNA3, {0xd8500000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_RDNA3,
       {0xf4000100u, 0xf8000000u},
       WaitCounterKind::Ds}, // s_load_b32 s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_RDNA3,
       {0xdc520000u, 0x27c0000u},
       WaitCounterKind::Load}, // global_load_b32 v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_RDNA3,
       {0xdc6a0000u, 0x7c0200u},
       WaitCounterKind::Store}, // global_store_b32 v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_RDNA3, {0xbe844c80u}, WaitCounterKind::Ds}, // s_sendmsg_rtn_b32 s4, 0x80
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       {0xd8d40000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       {0xdacc0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_RDNA3_5, {0xd8500000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       {0xf4000100u, 0xf8000000u},
       WaitCounterKind::Ds}, // s_load_b32 s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       {0xdc520000u, 0x27c0000u},
       WaitCounterKind::Load}, // global_load_b32 v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       {0xdc6a0000u, 0x7c0200u},
       WaitCounterKind::Store}, // global_store_b32 v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_RDNA3_5,
       {0xbe844c80u},
       WaitCounterKind::Ds}, // s_sendmsg_rtn_b32 s4, 0x80
      {ROCJITSU_CODE_ARCH_RDNA4,
       {0xd8d40000u, 0x2000001u},
       WaitCounterKind::Ds}, // ds_swizzle_b32 v2, v1
      {ROCJITSU_CODE_ARCH_RDNA4,
       {0xdacc0000u, 0x2000100u},
       WaitCounterKind::Ds}, // ds_bpermute_b32 v2, v0, v1
      {ROCJITSU_CODE_ARCH_RDNA4, {0xd8500000u, 0x0u}, WaitCounterKind::Ds}, // ds_nop
      {ROCJITSU_CODE_ARCH_RDNA4,
       {0xf4000100u, 0xf8000000u},
       WaitCounterKind::Km}, // s_load_b32 s4, s[0:1], 0
      {ROCJITSU_CODE_ARCH_RDNA4,
       {0xee05007cu, 0x2u, 0x0u},
       WaitCounterKind::Load}, // global_load_b32 v2, v[0:1], off
      {ROCJITSU_CODE_ARCH_RDNA4,
       {0xee06807cu, 0x1000000u, 0x0u},
       WaitCounterKind::Store}, // global_store_b32 v[0:1], v2, off
      {ROCJITSU_CODE_ARCH_RDNA4, {0xbe844c80u}, WaitCounterKind::Km}, // s_sendmsg_rtn_b32 s4, 0x80
  };
  for (const auto &c : cases) {
    SCOPED_TRACE(static_cast<int>(c.arch));
    auto inst = decode(c.words, c.arch);
    ASSERT_NE(inst, nullptr);
    SCOPED_TRACE(inst->mnemonic());
    EXPECT_TRUE(inst->is_memory_wait_producer());
    auto events = Target::classify_events(*inst, c.arch);
    ASSERT_TRUE(events.succeeded());
    ASSERT_FALSE(events.value().empty());
    EXPECT_EQ(events.value().front().counter, c.counter);
  }
}

TEST(WaitcheckTarget, PrefetchesDoNotContributeCompletionEntries) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                    ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
                    ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                    ROCJITSU_CODE_ARCH_RDNA4})
    for (auto name : {"s_prefetch_inst", "s_prefetch_data", "s_buffer_prefetch_data",
                      "global_prefetch_b8", "flat_prefetch_b8"}) {
      NamedInstruction inst(name);
      auto events = Target::classify_events(inst, arch);
      ASSERT_TRUE(events.succeeded());
      EXPECT_TRUE(events.value().empty()) << name;
    }
}

TEST(WaitcheckTarget, ImageQueriesUseTheirLoadOrSampleQueue) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
                    ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    NamedInstruction resinfo("image_get_resinfo");
    auto loads = Target::classify_events(resinfo, arch);
    ASSERT_TRUE(loads.succeeded());
    ASSERT_FALSE(loads.value().empty());
    EXPECT_EQ(loads.value()[0].counter, WaitCounterKind::Load);
    NamedInstruction lod("image_get_lod");
    auto samples = Target::classify_events(lod, arch);
    ASSERT_TRUE(samples.succeeded());
    ASSERT_FALSE(samples.value().empty());
    EXPECT_EQ(samples.value()[0].counter,
              arch == ROCJITSU_CODE_ARCH_RDNA4 ? WaitCounterKind::Sample : WaitCounterKind::Load);
  }
}

TEST(WaitcheckTarget, Gfx10LegacyLgkmUsesAllSixBits) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const std::array<uint32_t, 1> words{0xbf8ce07fu}; // vmcnt(63), expcnt(7), lgkmcnt(32).
    auto inst = decode(words, arch);
    ASSERT_NE(inst, nullptr);
    auto fields = Target::explicit_wait_fields(*inst, arch);
    ASSERT_TRUE(fields.succeeded());
    ASSERT_TRUE(fields.value());
    EXPECT_EQ((*fields.value())[Target::counter_index(WaitCounterKind::Ds)], 32u);
    EXPECT_FALSE((*fields.value())[Target::counter_index(WaitCounterKind::Load)]);
    EXPECT_EQ(Target::counter_no_wait_value(arch, WaitCounterKind::Ds).value(), 63u);
  }
}

TEST(WaitcheckTarget, OlderInterpolationHasNoEmbeddedExportWait) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    // v_interp_p1_f32 v0, v0, attr0.x, with architecture-specific format bits.
    const std::array<uint32_t, 1> words{arch == ROCJITSU_CODE_ARCH_CDNA1 ? 0xd4000000u
                                                                         : 0xc8000000u};
    auto inst = decode(words, arch);
    ASSERT_NE(inst, nullptr);
    auto fields = Target::embedded_wait_fields(*inst, arch);
    ASSERT_TRUE(fields.succeeded());
    EXPECT_FALSE(fields.value());
  }
}

TEST(WaitcheckTarget, LegacyExportAndMessagesContributeEvents) {
  const std::array<uint32_t, 2> exp_words{0xf800080fu, 0x03020100u};
  std::unique_ptr<Instruction> exp = decode(exp_words, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(exp, nullptr);
  auto events_result = Target::classify_events(*exp, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(events_result.succeeded());
  const auto events = std::move(events_result).value();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events[0].counter, WaitCounterKind::Exp);
  EXPECT_TRUE(events[0].check_exec_defs);
  for (std::string_view name : {"s_sendmsg", "s_sendmsghalt"}) {
    NamedInstruction message(name);
    auto messages_result = Target::classify_events(message, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(messages_result.succeeded());
    const auto messages = std::move(messages_result).value();
    ASSERT_EQ(messages.size(), 1u);
    EXPECT_EQ(messages[0].counter, WaitCounterKind::Ds);
    EXPECT_EQ(messages[0].kind, WaitEventKind::SqMessage);
    EXPECT_EQ(messages[0].registers, TrackedRegisterSource::None);
  }
}

TEST(WaitcheckTarget, FlatStoresKeepTheirIdentityOnBothCounters) {
  NamedInstruction inst("flat_store_dword");
  auto events_result = Target::classify_events(inst, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(events_result.succeeded());
  const auto events = std::move(events_result).value();
  ASSERT_GE(events.size(), 2u);
  EXPECT_EQ(events[0].kind, WaitEventKind::FlatStore);
  EXPECT_EQ(events[1].counter, WaitCounterKind::Ds);
  EXPECT_EQ(events[1].kind, WaitEventKind::FlatStore);
}

TEST(WaitcheckTarget, DirectToLdsIsNotAVgprResultOnBothCdnaTargets) {
  const std::array<uint32_t, 2> words{0xe05d1000u, 0x80100008u};
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    std::unique_ptr<Instruction> inst = decode(words, arch);
    ASSERT_NE(inst, nullptr);
    auto events_result = Target::classify_events(*inst, arch);
    ASSERT_TRUE(events_result.succeeded());
    const auto events = std::move(events_result).value();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].kind, WaitEventKind::LdsDirect);
    EXPECT_EQ(events[0].registers, TrackedRegisterSource::None);
    EXPECT_TRUE(events[0].check_counter_parity_order);
  }
}

TEST(WaitcheckTarget, GdsProtectsAddressAndDataSourcesUntilExpcnt) {
  for (const std::array<uint32_t, 2> words :
       {std::array<uint32_t, 2>{0xd8da0000u, 1u},        // ds_load_b32 v0, v1 gds.
        std::array<uint32_t, 2>{0xd8360000u, 0x201u},    // ds_store_b32 v1, v2 gds.
        std::array<uint32_t, 2>{0xd8020000u, 0x201u}}) { // ds_add_u32 v1, v2 gds.
    std::unique_ptr<Instruction> inst = decode(words, ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_NE(inst, nullptr);
    auto events_result = Target::classify_events(*inst, ROCJITSU_CODE_ARCH_RDNA3);
    ASSERT_TRUE(events_result.succeeded());
    const auto events = std::move(events_result).value();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].counter, WaitCounterKind::Ds);
    EXPECT_EQ(events[0].kind, WaitEventKind::Gds);
    EXPECT_EQ(events[1].counter, WaitCounterKind::Exp);
    EXPECT_EQ(events[1].registers, TrackedRegisterSource::VectorUses);
    EXPECT_TRUE(events[1].check_defs);
  }
  NamedInstruction always_gds("ds_ordered_count");
  EXPECT_EQ(Target::classify_events(always_gds, ROCJITSU_CODE_ARCH_CDNA4).value().size(), 2u);
}

TEST(WaitcheckTarget, TimestampAndBarrierStateHaveScalarResults) {
  const std::array<uint32_t, 2> timestamp{0xc0900000u, 0u};
  std::unique_ptr<Instruction> legacy = decode(timestamp, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(legacy, nullptr);
  auto old_events_result = Target::classify_events(*legacy, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_TRUE(old_events_result.succeeded());
  const auto old_events = std::move(old_events_result).value();
  ASSERT_FALSE(old_events.empty());
  EXPECT_EQ(old_events[0].counter, WaitCounterKind::Ds);
  EXPECT_EQ(old_events[0].registers, TrackedRegisterSource::Defs);
  const std::array<uint32_t, 1> barrier{0xbe805080u};
  std::unique_ptr<Instruction> modern = decode(barrier, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(modern, nullptr);
  auto new_events_result = Target::classify_events(*modern, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(new_events_result.succeeded());
  const auto new_events = std::move(new_events_result).value();
  ASSERT_EQ(new_events.size(), 1u);
  EXPECT_EQ(new_events[0].counter, WaitCounterKind::Km);
  EXPECT_EQ(new_events[0].registers, TrackedRegisterSource::Defs);
}

TEST(WaitcheckTarget, AluWaitIncludesVectorDestinationField) {
  const std::array<uint32_t, 1> words{0xbf880fffu};
  std::unique_ptr<Instruction> inst = decode(words, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(inst, nullptr);
  auto fields_result = Target::explicit_wait_fields(*inst, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(fields_result.succeeded());
  const auto fields = std::move(fields_result).value();
  ASSERT_TRUE(fields);
  EXPECT_EQ((*fields)[Target::counter_index(WaitCounterKind::VaVdst)], 0u);
  EXPECT_FALSE((*fields)[Target::counter_index(WaitCounterKind::VmVsrc)]);
}

} // namespace
} // namespace rocjitsu::waitcheck_detail
