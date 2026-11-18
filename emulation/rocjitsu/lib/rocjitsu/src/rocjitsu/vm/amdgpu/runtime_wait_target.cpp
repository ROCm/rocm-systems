// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/runtime_wait_target.h"

#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <cstdint>
#include <ranges>

namespace rocjitsu::waitcheck_detail {

util::FailureOr<std::vector<ClassifiedEvent>>
RuntimeWaitTarget::classify_events(const Instruction &inst, rj_code_arch_t arch) {
  std::vector<ClassifiedEvent> events;
  if (classify_events_into(inst, arch, events).failed())
    return util::Result::failure();
  return events;
}

util::Result RuntimeWaitTarget::classify_events_into(const Instruction &inst, rj_code_arch_t arch,
                                                     std::vector<ClassifiedEvent> &events) {
  if (WaitcheckTarget::classify_events_into(inst, arch, events).failed())
    return util::Result::failure();
  const auto mnemonic = inst.mnemonic();
  if (is_async_lds_load(mnemonic) || is_async_lds_store(mnemonic)) {
    const auto kind =
        is_async_lds_load(mnemonic) ? WaitEventKind::AsyncLdsLoad : WaitEventKind::AsyncLdsStore;
    events.clear();
    // Transfer completion belongs to ASYNCcnt. Translation remains replayable
    // until X drains, but it does not take a second VMEM completion position.
    events.emplace_back(WaitCounterKind::Async, kind, TrackedRegisterSource::None,
                        /*check_uses=*/false, /*check_defs=*/false,
                        /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/true, /*check_program_end=*/false);
    if (arch == ROCJITSU_CODE_ARCH_CDNA5)
      events.emplace_back(WaitCounterKind::X, kind, TrackedRegisterSource::Uses,
                          /*check_uses=*/false, /*check_defs=*/true, /*check_exec_defs=*/true);
  } else if (mnemonic == "buffer_wbl2") {
    for (auto &event : events)
      if (event.counter == WaitCounterKind::Load && event.kind == WaitEventKind::GlobalInv)
        event.kind = WaitEventKind::GlobalWb;
  } else if (arch == ROCJITSU_CODE_ARCH_CDNA5 && is_scalar_prefetch(mnemonic)) {
    // Scalar prefetch translates an address but never takes a KMcnt position.
    events.clear();
    events.emplace_back(WaitCounterKind::X, WaitEventKind::Smem, TrackedRegisterSource::Uses,
                        /*check_uses=*/false, /*check_defs=*/true,
                        /*check_exec_defs=*/false);
  }
  return util::Result::success();
}

std::optional<WaitEventKind>
RuntimeWaitTarget::normalized_hardware_event_kind(WaitCounterKind counter, WaitEventKind kind,
                                                  WaitcntModel model) {
  if (counter == WaitCounterKind::Load) {
    if (kind == WaitEventKind::GlobalInv ||
        (kind == WaitEventKind::GlobalWb && model == WaitcntModel::LegacyNoVscnt))
      return WaitEventKind::VmemNoSamplerLoad;
    if (kind == WaitEventKind::GlobalWb)
      return std::nullopt;
  }
  if (counter == WaitCounterKind::X && is_async_transfer_kind(kind))
    return WaitEventKind::VmemNoSamplerLoad;
  return WaitcheckTarget::normalized_hardware_event_kind(counter, kind, model);
}

std::optional<uint32_t> RuntimeWaitTarget::completion_order_kind(WaitCounterKind counter,
                                                                 WaitEventKind event_kind,
                                                                 rj_code_arch_t arch) {
  const auto model = amdgpu::waitcnt_model(arch).value();
  const bool legacy_flat =
      model == WaitcntModel::LegacyNoVscnt &&
      (event_kind == WaitEventKind::FlatLoad || event_kind == WaitEventKind::FlatStore);
  if (event_kind == WaitEventKind::Gds) {
    if (counter == WaitCounterKind::Ds && amdgpu::has_ordered_gds_writeback(arch))
      return uint32_t{1} << static_cast<unsigned>(WaitEventKind::Gds);
    return std::nullopt;
  }
  if (legacy_flat || event_kind == WaitEventKind::Smem ||
      event_kind == WaitEventKind::Export || event_kind == WaitEventKind::SqMessage ||
      event_kind == WaitEventKind::SccWrite ||
      (event_kind == WaitEventKind::GlobalWb && counter == WaitCounterKind::Load &&
       model != WaitcntModel::LegacyNoVscnt) ||
      event_kind == WaitEventKind::Unknown)
    return std::nullopt;
  uint32_t kind = 0;
  if (!(model == WaitcntModel::LegacyNoVscnt && counter == WaitCounterKind::Load))
    if (auto normalized = normalized_hardware_event_kind(counter, event_kind, model))
      kind = uint32_t{1} << static_cast<unsigned>(*normalized);
  if (counter == WaitCounterKind::Async)
    kind = uint32_t{1} << static_cast<unsigned>(event_kind == WaitEventKind::AsyncLdsStore
                                                    ? WaitEventKind::AsyncLdsStore
                                                    : WaitEventKind::AsyncLdsLoad);
  return kind;
}

bool RuntimeWaitTarget::flat_issues_both_counters(rj_code_arch_t arch) {
  return amdgpu::legacy_flat_issues_both_counter_domains(arch) ||
         amdgpu::flat_issues_both_routed_domains(arch);
}

uint32_t RuntimeWaitTarget::issue_units(const Instruction &inst, const ClassifiedEvent &event,
                                        rj_code_arch_t arch) {
  if (event.counter == WaitCounterKind::VaVdst && inst.mnemonic().starts_with("v_wmma_scale"))
    return 2;
  const auto model = amdgpu::waitcnt_model(arch).value();
  if (const auto *info = inst.amdgpu_memory_issue_info())
    for (const auto &obligation : info->counter_obligations())
      if (amdgpu::canonical_wait_counter(obligation.wait_counter_type(), model) == event.counter)
        return obligation.counter_increment();
  if (event.kind == WaitEventKind::SqMessage && inst.mnemonic().starts_with("s_sendmsg_rtn_"))
    return 2;
  return 1;
}

bool RuntimeWaitTarget::is_async_transfer_kind(WaitEventKind kind) {
  return kind == WaitEventKind::AsyncLdsLoad || kind == WaitEventKind::AsyncLdsStore;
}

bool RuntimeWaitTarget::releases_gfx10_scalar_sources(const Instruction &inst,
                                                      rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 &&
         ((inst.flags() & VALU) || (inst.mnemonic() == "s_waitcnt" && inst.src_operand(0) &&
                                    inst.src_operand(0)->encoding_value() == 0));
}

bool RuntimeWaitTarget::export_skips_with_empty_exec(const Instruction &inst, rj_code_arch_t arch,
                                                     uint64_t exec) {
  if (exec || !amdgpu::has_typed_export_waits(arch))
    return false;
  amdgpu::RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  const auto &export_operands = modifiers.shader_export;
  return export_operands && !export_operands->done &&
         export_operands->target != amdgpu::ShaderExportOperands::kFirstPosition;
}

std::optional<ClassifiedEvent> RuntimeWaitTarget::gfx10_scalar_source(const Instruction &inst,
                                                                      rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA1 || !(inst.flags() & (VMEM | LDS)))
    return std::nullopt;
  const auto mnemonic = inst.mnemonic();
  const bool gds =
      (inst.flags() & LDS) && inst.raw_encoding() && (inst.raw_encoding()[0] & (1u << 17)) != 0;
  // GWS sources are held by EXPCNT rather than the native LDS source path.
  if (mnemonic.starts_with("ds_gws_"))
    return std::nullopt;
  const bool uses_m0 = gds || mnemonic.find("_addtid_") != std::string_view::npos ||
                       mnemonic == "ds_append" || mnemonic == "ds_consume" ||
                       mnemonic == "ds_ordered_count";
  return ClassifiedEvent(WaitCounterKind::VmVsrc, kGfx10ScalarSource, TrackedRegisterSource::Uses,
                         /*check_uses=*/false, /*check_defs=*/true,
                         /*check_exec_defs=*/true,
                         uses_m0 ? std::optional{RegisterRef{RegClass::M0, 0, 1}} : std::nullopt);
}

uint8_t RuntimeWaitTarget::wmma_valu_wait_states(const Instruction &inst, rj_code_arch_t arch,
                                                 rj_code_target_id_t target) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5)
    return 0;
  const auto name = inst.mnemonic();
  if (name.starts_with("v_wmma_") &&
      (name.ends_with("_16x16x32_f16") || name.ends_with("_16x16x32_bf16")))
    return target == ROCJITSU_CODE_TARGET_GFX1250 ? 4 : 8;
  return 0;
}

std::optional<WaitEventKind> RuntimeWaitTarget::expert_valu_kind(const Instruction &inst,
                                                                 rj_code_arch_t arch) {
  if (!amdgpu::supports_expert_scheduling(arch) || !(inst.flags() & VALU) || inst.is_memory_op())
    return std::nullopt;
  const std::string_view op = inst.mnemonic();
  const bool matrix = op.starts_with("v_wmma_") || op.starts_with("v_swmmac_");
  const bool vector_matrix = arch == ROCJITSU_CODE_ARCH_CDNA5 &&
                             (op == "v_wmma_f32_16x16x4_f32" || op == "v_wmma_f64_16x16x4_f64");
  if (op.starts_with("v_dot") || (matrix && !vector_matrix))
    return kValuMatrix;
  if (inst.is_transcendental())
    return kValuTrans;
  static constexpr std::string_view dp[] = {
      "v_add_f64",           "v_ashr_i64",       "v_ceil_f64",       "v_cvt_f32_f64",
      "v_cvt_i32_f64",       "v_cvt_u32_f64",    "v_div_fixup_f64",  "v_div_fmas_f64",
      "v_div_scale_f64",     "v_floor_f64",      "v_fma_f64",        "v_fract_f64",
      "v_frexp_exp_i32_f64", "v_frexp_mant_f64", "v_ldexp_f64",      "v_lshl_b64",
      "v_lshr_b64",          "v_maximum_f64",    "v_max_num_f64",    "v_minimum_f64",
      "v_min_num_f64",       "v_mul_f64",        "v_rcp_f64",        "v_rndne_f64",
      "v_rsq_f64",           "v_sqrt_f64",       "v_trig_preop_f64", "v_trunc_f64"};
  const auto stem = op.substr(0, std::min({op.find("_e32"), op.find("_e64"), op.find("_dpp")}));
  if (std::ranges::find(dp, stem) != std::end(dp))
    return kValuDp;
  // LLVM groups F64 and 64-bit integer VOPC/VOPCX compares with DPMACC.
  // The class-F64 forms are separate until LLVM assigns them that class.
  const bool compare = stem.starts_with("v_cmp_") || stem.starts_with("v_cmpx_");
  const bool class_f64 = stem == "v_cmp_class_f64" || stem == "v_cmpx_class_f64";
  if (compare && !class_f64 &&
      (stem.ends_with("_f64") || stem.ends_with("_i64") || stem.ends_with("_u64")))
    return kValuDp;
  return kValuCore;
}

bool RuntimeWaitTarget::expert_valu_issues_without_active_lanes(const Instruction &inst,
                                                                rj_code_arch_t arch,
                                                                unsigned wave_size) {
  RegisterSet defs;
  inst.implicit_defs(defs);
  for (int i = 0; i < inst.num_dst_operands(); ++i)
    if (const auto ref = inst.dst_operand(i)->to_register_ref())
      defs.expand(*ref);
  const auto name = inst.mnemonic();
  // RDNA3/3.5/4 wave64 V_CMPX can skip one pass, but never both. It therefore
  // advances VA_VDST even with no active lanes; wave32 has no such guarantee.
  const bool cmpx_wave64 = wave_size == 64 && name.starts_with("v_cmpx_") &&
                           (arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                            arch == ROCJITSU_CODE_ARCH_RDNA3_5 || arch == ROCJITSU_CODE_ARCH_RDNA4);
  // CDNA5 and RDNA4 ISA 3.2.2.1: a plain VALU may be skipped with EXEC zero,
  // while scalar/VCC writers, matrix operations, and named controls must issue.
  return cmpx_wave64 || defs.intersects({RegClass::SGPR, 0, REGISTER_SET_MAX_SGPRS}) ||
         defs.contains({RegClass::VCC, 0, 1}) || name.starts_with("v_cmp_") ||
         name == "v_add_co_ci_u32_e32" || name == "v_sub_co_ci_u32_e32" ||
         name == "v_subrev_co_ci_u32_e32" || name.starts_with("v_wmma_") ||
         name.starts_with("v_swmmac_") || name.starts_with("v_nop") ||
         name.starts_with("v_pipeflush") || name.starts_with("v_readlane") ||
         name.starts_with("v_readfirstlane") || name.starts_with("v_writelane");
}

} // namespace rocjitsu::waitcheck_detail
