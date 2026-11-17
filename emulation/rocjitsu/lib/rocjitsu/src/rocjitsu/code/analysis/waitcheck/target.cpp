// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/target.h"

#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu {
namespace waitcheck_detail {

std::optional<RegisterRef> WaitcheckTarget::register_operand(const Operand &operand,
                                                             rj_code_arch_t arch) {
  if (const auto ref = operand.to_register_ref())
    return ref;
  if (operand.has_register_selector()) {
    const auto selector = static_cast<uint32_t>(operand.encoding_value());
    if (selector == amdgpu::kVccSelectorFirst || selector == amdgpu::kVccSelectorLast) {
      const auto half = static_cast<uint16_t>(selector - amdgpu::kVccSelectorFirst);
      const uint8_t width = half == 0 && operand.size_bits() >= 64 ? 2 : 1;
      return RegisterRef{RegClass::VCC, half, width};
    }
  }
  // Mesh exports read M0 through a fieldless input. Scalar instructions such
  // as S_SENDMSG_RTN_B32 can produce the same register through a selector.
  if ((amdgpu::has_typed_export_waits(arch) && operand.to_special_reg_class() == RegClass::M0) ||
      (operand.has_register_selector() &&
       static_cast<uint32_t>(operand.encoding_value()) == amdgpu::scalar_m0_selector(arch)))
    return RegisterRef{RegClass::M0, 0, 1};
  return std::nullopt;
}

std::optional<uint32_t> WaitcheckTarget::completion_order_kind(WaitCounterKind counter,
                                                               WaitEventKind event_kind,
                                                               rj_code_arch_t arch) {
  const auto model = amdgpu::waitcnt_model(arch).value();
  const bool legacy_flat =
      model == WaitcntModel::LegacyNoVscnt &&
      (event_kind == WaitEventKind::FlatLoad || event_kind == WaitEventKind::FlatStore);
  if (legacy_flat || event_kind == WaitEventKind::Smem || event_kind == WaitEventKind::Gds ||
      event_kind == WaitEventKind::Export || event_kind == WaitEventKind::SqMessage ||
      event_kind == WaitEventKind::SccWrite ||
      (event_kind == WaitEventKind::GlobalWb && counter == WaitCounterKind::Load &&
       model != WaitcntModel::LegacyNoVscnt) ||
      event_kind == WaitEventKind::Unknown)
    return std::nullopt;
  uint32_t kind = 0;
  if (!(model == WaitcntModel::LegacyNoVscnt && counter == WaitCounterKind::Load))
    if (auto normalized =
            WaitcheckTarget::normalized_hardware_event_kind(counter, event_kind, model))
      kind = uint32_t{1} << static_cast<unsigned>(*normalized);
  // ASYNC loads and stores share a counter but return done out of order with
  // respect to each other (CDNA5 ISA 10.8). Barrier arrive orders with loads.
  if (counter == WaitCounterKind::Async)
    kind = uint32_t{1} << static_cast<unsigned>(event_kind == WaitEventKind::AsyncLdsStore
                                                    ? WaitEventKind::AsyncLdsStore
                                                    : WaitEventKind::AsyncLdsLoad);
  // Older RDNA samples and BVH operations share VMcnt, not a completion FIFO.
  if (model == WaitcntModel::LegacyVscnt && counter == WaitCounterKind::Load &&
      (event_kind == WaitEventKind::Sample || event_kind == WaitEventKind::Bvh))
    kind = uint32_t{1} << static_cast<unsigned>(event_kind);
  return kind;
}

bool WaitcheckTarget::flat_issues_both_counters(rj_code_arch_t arch) {
  // Legacy generic FLAT issues both counters. GFX9's independent completions
  // are unordered; completion_order_kind handles that.
  return amdgpu::legacy_flat_issues_both_counter_domains(arch) ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

uint32_t WaitcheckTarget::issue_units(const Instruction &inst, const ClassifiedEvent &event,
                                      rj_code_arch_t arch) {
  if (event.counter == WaitCounterKind::VaVdst &&
      event.registers == TrackedRegisterSource::VectorUses)
    return 0;
  // A scaled VOP3PX2 WMMA reserves two positions in the one VA_VDST counter.
  if (event.counter == WaitCounterKind::VaVdst && inst.mnemonic().starts_with("v_wmma_scale"))
    return 2;
  const auto model = amdgpu::waitcnt_model(arch).value();

  if (const auto *info = inst.amdgpu_memory_issue_info())
    for (const auto &obligation : info->counter_obligations())
      if (amdgpu::canonical_wait_counter(obligation.wait_counter_type(), model) == event.counter)
        return obligation.counter_increment();
  // Inline message returns are outside the memory-pipeline metadata: one
  // completion acknowledges the send, the other the returned register value.
  if (event.kind == WaitEventKind::SqMessage && inst.mnemonic().starts_with("s_sendmsg_rtn_"))
    return 2;
  return 1;
}

bool WaitcheckTarget::vm_vsrc_event_implied_by_wait(WaitEventKind kind, WaitCounterKind counter) {
  switch (counter) {
  case WaitCounterKind::Load:
    return kind == WaitEventKind::VmemNoSamplerLoad || kind == WaitEventKind::FlatLoad;
  case WaitCounterKind::Store:
    return kind == WaitEventKind::VmemStore || kind == WaitEventKind::FlatStore;
  case WaitCounterKind::Ds:
    return kind == WaitEventKind::Ds || kind == WaitEventKind::FlatLoad ||
           kind == WaitEventKind::FlatStore;
  case WaitCounterKind::Sample:
    return kind == WaitEventKind::Sample;
  case WaitCounterKind::Bvh:
    return kind == WaitEventKind::Bvh;
  default:
    return false;
  }
}

std::optional<WaitEventKind>
WaitcheckTarget::normalized_hardware_event_kind(WaitCounterKind counter, WaitEventKind kind,
                                                WaitcntModel model) {
  switch (counter) {
  case WaitCounterKind::Load:
    // Generic FLAT and ordinary VMEM loads both raise VMEM_READ_ACCESS.
    // BUFFER_INV and legacy VMcnt BUFFER_WBL2 acknowledgments complete in
    // issue order with VMEM reads. GFX12 GLOBAL_WB uses Storecnt instead.
    if (kind == WaitEventKind::GlobalWb && model != WaitcntModel::LegacyNoVscnt)
      return std::nullopt;
    if (kind == WaitEventKind::GlobalInv || kind == WaitEventKind::GlobalWb ||
        kind == WaitEventKind::FlatLoad ||
        kind == WaitEventKind::LdsDirect ||
        (amdgpu::uses_legacy_waitcnt(model) &&
         (kind == WaitEventKind::Sample || kind == WaitEventKind::Bvh))) {
      return WaitEventKind::VmemNoSamplerLoad;
    }
    return kind;
  case WaitCounterKind::Ds:
    // A generic FLAT access raises the same LDS_ACCESS event as native DS.
    if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::FlatStore)
      return WaitEventKind::Ds;
    return kind;
  case WaitCounterKind::Store:
    // Generic FLAT and ordinary VMEM stores share the store-completion event.
    if (kind == WaitEventKind::GlobalWb || kind == WaitEventKind::FlatStore)
      return WaitEventKind::VmemStore;
    return kind;
  case WaitCounterKind::X:
    // X_CNT distinguishes VMEM_GROUP from SMEM_GROUP, not the underlying
    // load/store/image operation.
    if (kind == WaitEventKind::Smem)
      return WaitEventKind::Smem;
    if (is_xcnt_vmem_kind(kind) || is_async_transfer_kind(kind))
      return WaitEventKind::VmemNoSamplerLoad;
    return kind;
  case WaitCounterKind::VmVsrc:
    if (kind == WaitEventKind::Ds)
      return WaitEventKind::Ds;
    if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::FlatStore)
      return WaitEventKind::FlatLoad;
    if (is_xcnt_vmem_kind(kind))
      return WaitEventKind::VmemNoSamplerLoad;
    return kind;
  case WaitCounterKind::Async:
    // Load, store, and barrier forms all raise ASYNC_ACCESS.
    return WaitEventKind::AsyncLdsLoad;
  case WaitCounterKind::Tensor:
    return WaitEventKind::TensorLdsLoad;
  default:
    return kind;
  }
}

[[nodiscard]] util::FailureOr<std::optional<WaitFields>>
WaitcheckTarget::embedded_wait_fields(const Instruction &inst, rj_code_arch_t arch) {
  const auto model = amdgpu::waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  WaitFields fields;
  bool has_wait = false;
  const bool has_vinterp_wait = arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                                arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                                arch == ROCJITSU_CODE_ARCH_RDNA4;
  if (const auto wait_exp = vinterp_wait_exp(inst);
      has_vinterp_wait && wait_exp && *wait_exp < 0x7u) {
    fields[counter_index(WaitCounterKind::Exp)] = *wait_exp;
    has_wait = true;
  }
  if (const auto wait_vm_vsrc = dsdir_wait_vm_vsrc(inst);
      !amdgpu::uses_legacy_waitcnt(model.value()) && wait_vm_vsrc && *wait_vm_vsrc == 0) {
    fields[counter_index(WaitCounterKind::VmVsrc)] = 0;
    has_wait = true;
  }
  if (const auto wait_va_vdst = dsdir_wait_va_vdst(inst); wait_va_vdst && *wait_va_vdst < 0xfu) {
    fields[counter_index(WaitCounterKind::VaVdst)] = *wait_va_vdst;
    has_wait = true;
  }
  return has_wait ? std::optional{fields} : std::nullopt;
}

// LLVM GCNHazardRecognizer::fixVMEMtoScalarWriteHazards. A VALU or fully
// drained packed S_WAITCNT separates GFX10.1 vector-memory issue from a
// scalar overwrite. Neither is an emitted DEPCTR field for parity accounting.
bool WaitcheckTarget::releases_gfx10_scalar_sources(const Instruction &inst, rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 &&
         ((inst.flags() & VALU) || (inst.mnemonic() == "s_waitcnt" && inst.src_operand(0) &&
                                    inst.src_operand(0)->encoding_value() == 0));
}

bool WaitcheckTarget::checks_scalar_source(WaitEventKind kind, const Instruction &inst) {
  return kind != WaitEventKind::Gfx10ScalarSource || (inst.flags() & (SALU | SMEM));
}

bool WaitcheckTarget::is_xcnt_drain(const Instruction &inst) {
  return inst.flags() &
         (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR | XCNT_DRAIN);
}

bool WaitcheckTarget::is_async_transfer_kind(WaitEventKind kind) {
  return kind == WaitEventKind::AsyncLdsLoad || kind == WaitEventKind::AsyncLdsStore;
}

[[nodiscard]] bool WaitcheckTarget::is_xcnt_vmem_kind(WaitEventKind kind) {
  switch (kind) {
  case WaitEventKind::VmemNoSamplerLoad:
  case WaitEventKind::FlatLoad:
  case WaitEventKind::VmemStore:
  case WaitEventKind::FlatStore:
  case WaitEventKind::Sample:
  case WaitEventKind::Bvh:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] std::optional<int64_t> WaitcheckTarget::first_barrier_id(const Instruction &inst) {
  const Operand *op = inst.src_operand(0);
  if (!op)
    return std::nullopt;
  if (const auto value = op->const_value())
    return static_cast<int64_t>(*value);
  return static_cast<int64_t>(static_cast<int32_t>(op->encoding_value()));
}

[[nodiscard]] bool WaitcheckTarget::is_cdna_lds_dma_load(const Instruction &inst,
                                                         rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4)
    return false;
  const auto mnemonic = inst.mnemonic();
  if (mnemonic.starts_with("scratch_load_lds_") || mnemonic.starts_with("global_load_lds_"))
    return true;
  if (!mnemonic.starts_with("buffer_load") || inst.raw_encoding() == nullptr ||
      inst.size() < 2 * static_cast<int>(sizeof(uint32_t)))
    return false;

  constexpr uint32_t kMubufLdsBit = 1u << 16u;
  return (inst.raw_encoding()[0] & kMubufLdsBit) != 0;
}

[[nodiscard]] std::optional<uint32_t> WaitcheckTarget::vinterp_wait_exp(const Instruction &inst) {
  if (!inst.mnemonic().starts_with("v_interp_") || inst.raw_encoding() == nullptr ||
      inst.size() < static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  return (inst.raw_encoding()[0] >> 8u) & 0x7u;
}

[[nodiscard]] bool WaitcheckTarget::is_dsdir(std::string_view mnemonic) {
  return mnemonic == "ds_param_load" || mnemonic == "ds_direct_load" ||
         mnemonic == "lds_param_load" || mnemonic == "lds_direct_load";
}

[[nodiscard]] std::optional<uint32_t> WaitcheckTarget::dsdir_wait_va_vdst(const Instruction &inst) {
  if (!is_dsdir(inst.mnemonic()) || inst.raw_encoding() == nullptr ||
      inst.size() < static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  return (inst.raw_encoding()[0] >> 16u) & 0xFu;
}

[[nodiscard]] std::optional<uint32_t> WaitcheckTarget::dsdir_wait_vm_vsrc(const Instruction &inst) {
  const auto mnemonic = inst.mnemonic();
  if ((mnemonic != "ds_param_load" && mnemonic != "ds_direct_load") ||
      inst.raw_encoding() == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  return (inst.raw_encoding()[0] >> 23u) & 0x1u;
}

[[nodiscard]] bool WaitcheckTarget::is_scalar_memory_op(std::string_view mnemonic) {
  return mnemonic.starts_with("s_load") || mnemonic.starts_with("s_buffer_load") ||
         mnemonic.starts_with("s_store") || mnemonic.starts_with("s_buffer_store") ||
         mnemonic.starts_with("s_atc_probe") || mnemonic.starts_with("s_atomic_") ||
         mnemonic.starts_with("s_buffer_atomic_") || mnemonic.starts_with("s_scratch_load") ||
         mnemonic.starts_with("s_scratch_store") || mnemonic.starts_with("s_dcache_") ||
         mnemonic == "s_gl1_inv" || mnemonic == "s_memtime" || mnemonic == "s_memrealtime" ||
         mnemonic == "s_get_barrier_state" || mnemonic == "s_get_waveid_in_workgroup";
}

[[nodiscard]] bool WaitcheckTarget::is_vmem_store(std::string_view mnemonic) {
  return mnemonic.starts_with("global_store") || mnemonic.starts_with("scratch_store") ||
         mnemonic.starts_with("buffer_store") || mnemonic.starts_with("tbuffer_store") ||
         mnemonic.starts_with("image_store");
}

[[nodiscard]] bool WaitcheckTarget::is_vmem_atomic(std::string_view mnemonic) {
  return mnemonic.starts_with("global_atomic") || mnemonic.starts_with("flat_atomic") ||
         mnemonic.starts_with("buffer_atomic") || mnemonic.starts_with("image_atomic");
}

[[nodiscard]] bool WaitcheckTarget::is_image_atomic(std::string_view mnemonic) {
  return mnemonic.starts_with("image_atomic");
}

[[nodiscard]] bool WaitcheckTarget::uses_ds_wait_counter(std::string_view mnemonic) {
  // Match LLVM's TII.isDS && TII.usesLGKM_CNT classification. All VDS
  // instructions use DS_CNT except the gfx1250 async-barrier arrive, which
  // uses ASYNC_CNT instead. LDS-direct instructions have separate EXP_CNT
  // semantics below.
  return mnemonic.starts_with("ds_") && !is_dsdir(mnemonic) &&
         mnemonic != "ds_atomic_async_barrier_arrive_b64";
}

[[nodiscard]] bool WaitcheckTarget::is_async_lds_load(std::string_view mnemonic) {
  return mnemonic.starts_with("global_load_async_to_lds") ||
         mnemonic.starts_with("cluster_load_async_to_lds");
}

[[nodiscard]] bool WaitcheckTarget::is_async_lds_store(std::string_view mnemonic) {
  return mnemonic.starts_with("global_store_async_from_lds");
}

[[nodiscard]] bool WaitcheckTarget::is_tensor_lds_load(std::string_view mnemonic) {
  return mnemonic == "tensor_load_to_lds";
}

[[nodiscard]] bool WaitcheckTarget::is_tensor_lds_store(std::string_view mnemonic) {
  return mnemonic == "tensor_store_from_lds";
}

uint8_t WaitcheckTarget::wmma_valu_wait_states(const Instruction &inst, rj_code_arch_t arch,
                                               rj_code_target_id_t target) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA5)
    return 0;
  const auto name = inst.mnemonic();
  // CDNA5 ISA 7.12.1 and LLVM GCNHazardRecognizer's VALUWaitStates:
  // gfx1250's dense f16/bf16 WMMA needs four independent VALUs, gfx1251's
  // version needs eight. A completed later VALU proves progress even when
  // the receiving memory instruction can issue independently in expert mode.
  if (name.starts_with("v_wmma_") &&
      (name.ends_with("_16x16x32_f16") || name.ends_with("_16x16x32_bf16")))
    return target == ROCJITSU_CODE_TARGET_GFX1250 ? 4 : 8;
  return 0;
}

bool WaitcheckTarget::expert_valu_issues_without_active_lanes(const Instruction &inst,
                                                               const RegisterSet &defs,
                                                               rj_code_arch_t arch,
                                                               uint32_t wave_size) {
  const auto name = inst.mnemonic();
  // A wave64 V_CMPX on RDNA3/3.5/4 can skip one pass, but not both. A wave32
  // V_CMPX and CDNA5's EXEC-only form cannot prove a younger issue at EXEC=0.
  const bool cmpx_wave64 = wave_size == 64 && name.starts_with("v_cmpx_") &&
                           (arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                            arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                            arch == ROCJITSU_CODE_ARCH_RDNA4);
  return cmpx_wave64 || defs.intersects({RegClass::SGPR, 0, REGISTER_SET_MAX_SGPRS}) ||
         defs.contains({RegClass::VCC, 0, 1}) || name.starts_with("v_cmp_") ||
         name == "v_add_co_ci_u32_e32" || name == "v_sub_co_ci_u32_e32" ||
         name == "v_subrev_co_ci_u32_e32" || name.starts_with("v_wmma_") ||
         name.starts_with("v_swmmac_") || name.starts_with("v_nop") ||
         name.starts_with("v_pipeflush") || name.starts_with("v_readlane") ||
         name.starts_with("v_readfirstlane") || name.starts_with("v_writelane");
}

std::optional<WaitEventKind> WaitcheckTarget::expert_valu_kind(const Instruction &inst,
                                                               rj_code_arch_t arch) {
  if (!amdgpu::supports_expert_scheduling(arch) || !(inst.flags() & VALU) || inst.is_memory_op())
    return std::nullopt;
  const std::string_view op = inst.mnemonic();
  // SIInstrInfo::isXDL includes DOT. The CDNA5 f32/f64 16x16x4
  // matrix forms are normal VALU operations in LLVM's WMMAInstInfoTable.
  const bool matrix = op.starts_with("v_wmma_") || op.starts_with("v_swmmac_");
  const bool vector_matrix = arch == ROCJITSU_CODE_ARCH_CDNA5 &&
                             (op == "v_wmma_f32_16x16x4_f32" || op == "v_wmma_f64_16x16x4_f64");
  if (op.starts_with("v_dot") || (matrix && !vector_matrix))
    return WaitEventKind::ValuMatrix;
  if (inst.is_transcendental())
    return WaitEventKind::ValuTrans;
  // LLVM AMDGPUHWEvents separates the double-precision MACC pipe from
  // core/side VALUs. F64 conversions *from* 32-bit values are core operations.
  static constexpr std::string_view dp[] = {
      "v_add_f64",           "v_ashr_i64",       "v_ceil_f64",       "v_cvt_f32_f64",
      "v_cvt_i32_f64",       "v_cvt_u32_f64",    "v_div_fixup_f64",  "v_div_fmas_f64",
      "v_div_scale_f64",     "v_floor_f64",      "v_fma_f64",        "v_fract_f64",
      "v_frexp_exp_i32_f64", "v_frexp_mant_f64", "v_ldexp_f64",      "v_lshl_b64",
      "v_lshr_b64",          "v_maximum_f64",    "v_max_num_f64",    "v_minimum_f64",
      "v_min_num_f64",       "v_mul_f64",        "v_rcp_f64",        "v_rndne_f64",
      "v_rsq_f64",           "v_sqrt_f64",       "v_trig_preop_f64", "v_trunc_f64"};
  const auto stem = op.substr(0, std::min({op.find("_e32"), op.find("_e64"), op.find("_dpp")}));
  // LLVM tags the 64-bit VOPC/VOPCX comparisons as double-precision pipe
  // operations. Its CLASS_F64 forms do not carry that tag.
  const bool dp_compare =
      (stem.starts_with("v_cmp_") || stem.starts_with("v_cmpx_")) &&
      stem != "v_cmp_class_f64" && stem != "v_cmpx_class_f64" &&
      (stem.ends_with("_f64") || stem.ends_with("_i64") || stem.ends_with("_u64"));
  if (dp_compare || std::ranges::find(dp, stem) != std::end(dp))
    return WaitEventKind::ValuDp;
  return WaitEventKind::ValuCore;
}

[[nodiscard]] util::FailureOr<std::vector<ClassifiedEvent>>
WaitcheckTarget::classify_events(const Instruction &inst, rj_code_arch_t arch) {
  std::vector<ClassifiedEvent> events;
  if (classify_events_into(inst, arch, events).failed())
    return util::Result::failure();
  return events;
}

std::optional<ClassifiedEvent> WaitcheckTarget::gfx10_scalar_source(const Instruction &inst,
                                                                    rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA1 || !(inst.flags() & (VMEM | LDS)))
    return std::nullopt;
  const auto mnemonic = inst.mnemonic();
  // GFX10.1 needs a zero VM_VSRC dependency wait even in ordinary scheduling
  // mode. Vector-memory instructions read EXEC, in addition to their encoded
  // scalar inputs. GDS and the address-bearing DS forms below also read M0.
  const bool gds =
      (inst.flags() & LDS) && inst.raw_encoding() && (inst.raw_encoding()[0] & (1u << 17)) != 0;
  const bool uses_m0 = gds || mnemonic.find("_addtid_") != std::string_view::npos ||
                       mnemonic == "ds_append" || mnemonic == "ds_consume" ||
                       mnemonic == "ds_ordered_count" || mnemonic.starts_with("ds_gws_");
  return ClassifiedEvent(WaitCounterKind::VmVsrc, WaitEventKind::Gfx10ScalarSource,
                         TrackedRegisterSource::Uses, /*check_uses=*/false, /*check_defs=*/true,
                         /*check_exec_defs=*/true,
                         uses_m0 ? std::optional{RegisterRef{RegClass::M0, 0, 1}} : std::nullopt);
}

[[nodiscard]] util::Result
WaitcheckTarget::classify_events_into(const Instruction &inst, rj_code_arch_t arch,
                                      std::vector<ClassifiedEvent> &events) {
  if (classify_memory_events_into(inst, arch, events).failed())
    return util::Result::failure();
  if (auto source = gfx10_scalar_source(inst, arch))
    events.push_back(*source);
  if (const auto kind = expert_valu_kind(inst, arch)) {
    // Separate operand lifetimes share one hardware issue position. The read
    // facet carries no additional credits (see issue_units).
    events.emplace_back(WaitCounterKind::VaVdst, *kind, TrackedRegisterSource::VectorDefs);
    events.emplace_back(WaitCounterKind::VaVdst, *kind, TrackedRegisterSource::VectorUses,
                        /*check_uses=*/false, /*check_defs=*/true);
  }
  return util::Result::success();
}

util::Result WaitcheckTarget::classify_memory_events_into(const Instruction &inst,
                                                          rj_code_arch_t arch,
                                                          std::vector<ClassifiedEvent> &events) {
  events.clear();
  const auto model = amdgpu::waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  auto has_register_result = [&] {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      if (const Operand *operand = inst.dst_operand(i); operand && register_operand(*operand, arch))
        return true;
    }
    return false;
  };
  const bool expert = amdgpu::supports_expert_scheduling(arch);
  const auto mnemonic = inst.mnemonic();
  auto add_xcnt_event = [&](WaitEventKind kind) {
    if (arch == ROCJITSU_CODE_ARCH_CDNA5)
      events.emplace_back(WaitCounterKind::X, kind, TrackedRegisterSource::Uses,
                          /*check_uses=*/false, /*check_defs=*/true,
                          /*check_exec_defs=*/is_xcnt_vmem_kind(kind));
  };
  if (is_async_lds_load(mnemonic) || is_async_lds_store(mnemonic)) {
    const bool is_load = is_async_lds_load(mnemonic);
    const WaitEventKind kind = is_load ? WaitEventKind::AsyncLdsLoad : WaitEventKind::AsyncLdsStore;
    // CDNA5 10.8 assigns transfer completion to ASYNCcnt. LLVM omits the
    // translation event, but the ISA gives direct async VMEM no replay exemption.
    events.emplace_back(WaitCounterKind::Async, kind, TrackedRegisterSource::None,
                        /*check_uses=*/false, /*check_defs=*/false,
                        /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/true, /*check_program_end=*/false);
    if (arch == ROCJITSU_CODE_ARCH_CDNA5)
      events.emplace_back(WaitCounterKind::X, kind, TrackedRegisterSource::Uses,
                          /*check_uses=*/false, /*check_defs=*/true, /*check_exec_defs=*/true);
    // LLVM excludes ASYNC_CNT/TENSOR_CNT from blanket function-boundary
    // waits. Their waits are derived from object-invisible ASYNCMARK pairs,
    // so conservatively check observable LDS consumers but not program end.
    return util::Result::success();
  }

  if (mnemonic == "ds_atomic_async_barrier_arrive_b64") {
    events.emplace_back(WaitCounterKind::Async, WaitEventKind::AsyncBarrier,
                        TrackedRegisterSource::None, /*check_uses=*/false,
                        /*check_defs=*/false, /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/true,
                        /*check_program_end=*/false);
    return util::Result::success();
  }

  if (uses_ds_wait_counter(mnemonic)) {
    const uint32_t gds_bit = model.value() == WaitcntModel::LegacyNoVscnt ? 16u : 17u;
    const bool gds = amdgpu::uses_legacy_waitcnt(model.value()) &&
                     (mnemonic == "ds_ordered_count" || mnemonic == "ds_add_gs_reg_rtn" ||
                      mnemonic == "ds_sub_gs_reg_rtn" || mnemonic.starts_with("ds_gws_") ||
                      (inst.raw_encoding() != nullptr && inst.size() >= 8 &&
                       (inst.raw_encoding()[0] & (1u << gds_bit)) != 0));
    events.push_back({WaitCounterKind::Ds, gds ? WaitEventKind::Gds : WaitEventKind::Ds});
    // CDNA4 retains the legacy wait encoding, but EXPCNT is unused (ISA 3.1/4.4).
    if (gds && arch != ROCJITSU_CODE_ARCH_CDNA4)
      events.push_back({WaitCounterKind::Exp, WaitEventKind::Gds, TrackedRegisterSource::VectorUses,
                        false, true, true});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Ds,
                        TrackedRegisterSource::VectorUses, false, true});
    return util::Result::success();
  }

  if (is_tensor_lds_load(mnemonic) || is_tensor_lds_store(mnemonic)) {
    const bool is_load = is_tensor_lds_load(mnemonic);
    const WaitEventKind kind =
        is_load ? WaitEventKind::TensorLdsLoad : WaitEventKind::TensorLdsStore;
    events.emplace_back(WaitCounterKind::Tensor, kind, TrackedRegisterSource::None,
                        /*check_uses=*/false, /*check_defs=*/false,
                        /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/true, /*check_program_end=*/false);
    // AMDGPU::getEventsFor classifies tensor operations solely as
    // TENSOR_ACCESS, before the generic VMEM/X_CNT path.
    return util::Result::success();
  }

  if (mnemonic.starts_with("flat_load")) {
    events.push_back({WaitCounterKind::Load, WaitEventKind::FlatLoad});
    // Keep the instruction kind distinct: generic FLAT and native DS can
    // complete out of order even though both contribute to the DS counter.
    events.push_back({WaitCounterKind::Ds, WaitEventKind::FlatLoad});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::FlatLoad,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::FlatLoad);
    return util::Result::success();
  }

  if (mnemonic == "global_inv" || mnemonic == "buffer_inv" || mnemonic == "buffer_wbl2") {
    const auto kind = mnemonic == "buffer_wbl2" ? WaitEventKind::GlobalWb : WaitEventKind::GlobalInv;
    const bool legacy_wbl2 = mnemonic == "buffer_wbl2" &&
                             (arch == ROCJITSU_CODE_ARCH_CDNA2 ||
                              arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4);
    // Keep the writeback ACK in parity state so a following vmcnt(0) can be
    // attributed to its completion, even when no register payload is present.
    events.emplace_back(WaitCounterKind::Load, kind, TrackedRegisterSource::None, false, false,
                        false, std::nullopt, std::nullopt, false, false, legacy_wbl2);
    return util::Result::success();
  }

  if (mnemonic == "global_wb" || mnemonic == "global_wbinv") {
    events.emplace_back(amdgpu::vmem_store_wait_counter(model.value()), WaitEventKind::GlobalWb,
                        TrackedRegisterSource::None, false, false);
    return util::Result::success();
  }

  if (is_cdna_lds_dma_load(inst, arch)) {
    events.emplace_back(WaitCounterKind::Load, WaitEventKind::LdsDirect,
                        TrackedRegisterSource::None, /*check_uses=*/false,
                        /*check_defs=*/false, /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/false,
                        /*check_program_end=*/false,
                        /*check_counter_parity_order=*/true);
    return util::Result::success();
  }

  if (mnemonic.starts_with("global_load") || mnemonic.starts_with("scratch_load") ||
      mnemonic.starts_with("buffer_load") || mnemonic.starts_with("tbuffer_load") ||
      mnemonic.starts_with("image_load") || mnemonic == "image_get_resinfo") {
    events.push_back({WaitCounterKind::Load, WaitEventKind::VmemNoSamplerLoad});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemNoSamplerLoad,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::VmemNoSamplerLoad);
    return util::Result::success();
  }

  if (is_image_atomic(mnemonic)) {
    // Image models expose vdata as a destination even for no-return forms.
    // Read the return control until that distinction is available as metadata:
    // GFX9/10 GLC is bit 13, GFX11 GLC is bit 14, and GFX12 TH's
    // low bit is bit 20 of the second word.
    bool returns_value = has_register_result();
    if (inst.raw_encoding() != nullptr && inst.size() >= 8) {
      const unsigned glc_bit =
          arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ? 14u : 13u;
      returns_value = amdgpu::uses_legacy_waitcnt(model.value())
                          ? (inst.raw_encoding()[0] & (1u << glc_bit)) != 0
                          : (inst.raw_encoding()[1] & (1u << 20u)) != 0;
    }
    const WaitCounterKind counter =
        returns_value ? WaitCounterKind::Load : amdgpu::vmem_store_wait_counter(model.value());
    const WaitEventKind kind =
        returns_value ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore;
    events.emplace_back(counter, kind,
                        returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        /*check_uses=*/returns_value, /*check_defs=*/returns_value);
    if (expert)
      events.push_back(
          {WaitCounterKind::VmVsrc, kind, TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(kind);
    return util::Result::success();
  }

  if (is_vmem_atomic(mnemonic)) {
    const bool returns_value = has_register_result();
    const WaitCounterKind counter =
        returns_value ? WaitCounterKind::Load : amdgpu::vmem_store_wait_counter(model.value());
    const WaitEventKind kind =
        mnemonic.starts_with("flat_atomic")
            ? (returns_value ? WaitEventKind::FlatLoad : WaitEventKind::FlatStore)
            : (returns_value ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore);
    events.emplace_back(counter, kind,
                        returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        /*check_uses=*/returns_value, /*check_defs=*/returns_value);
    if (mnemonic.starts_with("flat_atomic"))
      events.emplace_back(WaitCounterKind::Ds, kind,
                          returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                          /*check_uses=*/returns_value, /*check_defs=*/returns_value);
    if (expert)
      events.emplace_back(WaitCounterKind::VmVsrc, kind, TrackedRegisterSource::VectorUses,
                          /*check_uses=*/false, /*check_defs=*/true);
    add_xcnt_event(kind);
    return util::Result::success();
  }

  if (mnemonic.starts_with("flat_store")) {
    events.push_back({amdgpu::vmem_store_wait_counter(model.value()), WaitEventKind::FlatStore,
                      TrackedRegisterSource::None, false, false});
    events.push_back(
        {WaitCounterKind::Ds, WaitEventKind::FlatStore, TrackedRegisterSource::None, false, false});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::FlatStore,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::FlatStore);
    return util::Result::success();
  }

  if (is_vmem_store(mnemonic)) {
    // The LDS-to-buffer transfer also occupies LGKM on targets whose issue metadata
    // includes it. Account for that unordered position even if a store fault cancels it.
    if (mnemonic == "buffer_store_lds_dword") {
      const auto *issue = inst.amdgpu_memory_issue_info();
      if (issue && std::ranges::any_of(issue->counter_obligations(), [](const auto &obligation) {
            return obligation.wait_counter_type() == amdgpu::WaitCounterType::LGKMCNT;
          }))
        events.emplace_back(WaitCounterKind::Ds, WaitEventKind::Unknown,
                            TrackedRegisterSource::None, false, false);
    }
    events.push_back({amdgpu::vmem_store_wait_counter(model.value()), WaitEventKind::VmemStore,
                      TrackedRegisterSource::None, false, false});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemStore,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::VmemStore);
    return util::Result::success();
  }

  if (mnemonic == "ds_param_load" || mnemonic == "ds_direct_load" || mnemonic == "lds_param_load" ||
      mnemonic == "lds_direct_load") {
    events.push_back({WaitCounterKind::Exp, WaitEventKind::LdsDirect});
    return util::Result::success();
  }

  // CDNA5 scalar prefetch reads an address through translation but never
  // increments KMCNT (ISA, scalar prefetch instructions).
  if (is_scalar_prefetch(mnemonic)) {
    add_xcnt_event(WaitEventKind::Smem);
    return util::Result::success();
  }

  if (is_scalar_memory_op(mnemonic)) {
    const bool produces_result = has_register_result();
    events.emplace_back(amdgpu::smem_wait_counter(model.value()), WaitEventKind::Smem,
                        produces_result ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        /*check_uses=*/produces_result, /*check_defs=*/produces_result);
    // Timestamp and barrier-state SOP operations account as SMEM_ACCESS,
    // but unlike SMRD instructions they do not translate a scalar address.
    if (mnemonic != "s_memtime" && mnemonic != "s_memrealtime" &&
        mnemonic != "s_get_barrier_state" && mnemonic != "s_get_waveid_in_workgroup")
      add_xcnt_event(WaitEventKind::Smem);
    return util::Result::success();
  }

  if (mnemonic == "s_sendmsg" || mnemonic == "s_sendmsghalt" || mnemonic == "s_sendmsg_rtn_b32" ||
      mnemonic == "s_sendmsg_rtn_b64") {
    const bool returns_value = has_register_result();
    events.emplace_back(amdgpu::smem_wait_counter(model.value()), WaitEventKind::SqMessage,
                        returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        returns_value, returns_value);
    return util::Result::success();
  }

  const auto vmem_type = amdgpu::vmem_read_type(arch, mnemonic);
  if (vmem_type == amdgpu::VmemWriteback::Sample ||
      vmem_type == amdgpu::VmemWriteback::PointSample) {
    events.push_back({amdgpu::image_sample_wait_counter(model.value()), WaitEventKind::Sample});
    events.back().point_sample_accel = vmem_type == amdgpu::VmemWriteback::PointSample;
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Sample,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::Sample);
    return util::Result::success();
  }

  if (vmem_type == amdgpu::VmemWriteback::Bvh) {
    events.push_back({amdgpu::image_bvh_wait_counter(model.value()), WaitEventKind::Bvh});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Bvh,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::Bvh);
    return util::Result::success();
  }

  if (mnemonic == "export" || mnemonic == "exp") {
    auto kind = WaitEventKind::Export;
    auto sources = TrackedRegisterSource::Uses;
    if (amdgpu::has_typed_export_waits(arch)) {
      // M0 supplies the row at issue. The export grant reads the enabled VGPRs
      // and EXEC; the latter is protected by check_exec_defs.
      sources = TrackedRegisterSource::VectorUses;
      amdgpu::RegisterModifiers modifiers;
      inst.amdgpu_register_modifiers(modifiers);
      if (modifiers.shader_export) {
        using Export = amdgpu::ShaderExportOperands;
        const uint8_t target = modifiers.shader_export->target;
        // RDNA4 groups color with depth but gives the blend pair a separate FIFO (ISA 5.7).
        if (target <= Export::kLastMrt)
          kind = WaitEventKind::ExportMrt;
        else if (target == Export::kDualSource0 || target == Export::kDualSource1)
          kind = arch == ROCJITSU_CODE_ARCH_RDNA4 ? WaitEventKind::ExportDualSourceBlend
                                                  : WaitEventKind::ExportMrt;
        else if (target == Export::kDepth)
          kind =
              arch == ROCJITSU_CODE_ARCH_RDNA4 ? WaitEventKind::ExportMrt : WaitEventKind::ExportZ;
        else if (target >= Export::kFirstPosition && target <= Export::kLastPosition)
          kind = WaitEventKind::ExportPosition;
        else if (target == Export::kPrimitive)
          kind = WaitEventKind::ExportPrimitive;
      }
    }
    events.push_back({WaitCounterKind::Exp, kind, sources, false, true, true});
    return util::Result::success();
  }

  if (mnemonic == "s_barrier_signal_isfirst") {
    if (!amdgpu::uses_legacy_waitcnt(model.value()))
      events.push_back({WaitCounterKind::Km, WaitEventKind::SccWrite, TrackedRegisterSource::None,
                        true, true, false, RegisterRef{RegClass::SCC, 0, 1},
                        first_barrier_id(inst)});
    return util::Result::success();
  }

  return util::Result::success();
}

} // namespace waitcheck_detail
} // namespace rocjitsu
