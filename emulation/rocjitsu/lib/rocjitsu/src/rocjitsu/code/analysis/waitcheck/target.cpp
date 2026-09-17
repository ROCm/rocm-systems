// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/target.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu {
std::string_view wait_counter_name(WaitCounterKind counter) {
  switch (counter) {
  case WaitCounterKind::Load:
    return "loadcnt";
  case WaitCounterKind::Store:
    return "storecnt";
  case WaitCounterKind::Ds:
    return "dscnt";
  case WaitCounterKind::Km:
    return "kmcnt";
  case WaitCounterKind::Sample:
    return "samplecnt";
  case WaitCounterKind::Bvh:
    return "bvhcnt";
  case WaitCounterKind::Exp:
    return "expcnt";
  case WaitCounterKind::X:
    return "xcnt";
  case WaitCounterKind::Async:
    return "asynccnt";
  case WaitCounterKind::Tensor:
    return "tensorcnt";
  case WaitCounterKind::VmVsrc:
    return "depctr_vm_vsrc";
  case WaitCounterKind::VaVdst:
    return "wait_va_vdst";
  case WaitCounterKind::Depctr:
    return "depctr";
  case WaitCounterKind::Count:
    break;
  }
  return "unknown";
}

namespace waitcheck_detail {

[[nodiscard]] size_t WaitcheckTarget::counter_index(WaitCounterKind counter) {
  return static_cast<size_t>(counter);
}

[[nodiscard]] util::FailureOr<uint32_t>
WaitcheckTarget::maximum_dependency_wait(rj_code_arch_t arch, WaitCounterKind counter) {
  const auto model = waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  // LLVM caps a dependency score at the largest non-sentinel wait value.
  // The all-ones encoding means "no wait", so the largest useful value is
  // one less than the hardware counter mask.
  switch (counter) {
  case WaitCounterKind::Load:
  case WaitCounterKind::Store:
    return 62;
  case WaitCounterKind::Ds:
    return uses_legacy_waitcnt(model.value()) && arch != ROCJITSU_CODE_ARCH_RDNA3 &&
                   arch != ROCJITSU_CODE_ARCH_RDNA3_5
               ? 14
               : 62;
  case WaitCounterKind::Km:
    return 30;
  case WaitCounterKind::Sample:
    return 62;
  case WaitCounterKind::Bvh:
  case WaitCounterKind::Exp:
  case WaitCounterKind::VmVsrc:
    return 6;
  case WaitCounterKind::X:
  case WaitCounterKind::Async:
  case WaitCounterKind::Tensor:
    return 62;
  case WaitCounterKind::VaVdst:
    return 14;
  case WaitCounterKind::Depctr:
  case WaitCounterKind::Count:
    return std::numeric_limits<uint32_t>::max();
  }
  return std::numeric_limits<uint32_t>::max();
}

[[nodiscard]] util::FailureOr<std::optional<uint32_t>>
WaitcheckTarget::counter_no_wait_value(rj_code_arch_t arch, WaitCounterKind counter) {
  const auto model = waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  if (uses_legacy_waitcnt(model.value())) {
    switch (counter) {
    case WaitCounterKind::Load:
      return 0x3fu;
    case WaitCounterKind::Store:
      return has_legacy_vscnt(model.value()) ? std::optional<uint32_t>{0x3fu} : std::nullopt;
    case WaitCounterKind::Ds:
      return arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ? 0x3fu : 0x0fu;
    case WaitCounterKind::Exp:
      return 0x07u;
    case WaitCounterKind::VmVsrc:
      return has_legacy_vscnt(model.value()) ? std::optional<uint32_t>{0x07u} : std::nullopt;
    case WaitCounterKind::VaVdst:
      return has_legacy_vscnt(model.value()) ? std::optional<uint32_t>{0x0fu} : std::nullopt;
    default:
      return std::nullopt;
    }
  }

  switch (counter) {
  case WaitCounterKind::Load:
  case WaitCounterKind::Store:
  case WaitCounterKind::Ds:
  case WaitCounterKind::Sample:
    return 0x3fu;
  case WaitCounterKind::X:
  case WaitCounterKind::Async:
  case WaitCounterKind::Tensor:
    return arch == ROCJITSU_CODE_ARCH_CDNA5 ? std::optional<uint32_t>{0x3fu} : std::nullopt;
  case WaitCounterKind::Km:
    return 0x1fu;
  case WaitCounterKind::Bvh:
  case WaitCounterKind::Exp:
  case WaitCounterKind::VmVsrc:
    return 0x07u;
  case WaitCounterKind::VaVdst:
    return 0x0fu;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] util::FailureOr<std::optional<WaitFields>>
WaitcheckTarget::explicit_wait_fields(const Instruction &inst, rj_code_arch_t arch) {
  const auto model = waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  WaitFields fields;
  auto operand_value = [&](int index) -> std::optional<uint32_t> {
    const Operand *op = inst.src_operand(index);
    if (!op)
      return std::nullopt;
    return static_cast<uint32_t>(op->encoding_value());
  };
  auto set_field = [&](WaitCounterKind counter, uint32_t value) {
    const auto no_wait = counter_no_wait_value(arch, counter).value();
    if (no_wait && value < *no_wait)
      fields[counter_index(counter)] = value;
  };

  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_wait_idle") {
    for (size_t counter_idx = 0; counter_idx < kCounterCount; ++counter_idx) {
      const auto counter = static_cast<WaitCounterKind>(counter_idx);
      if (counter_no_wait_value(arch, counter).value())
        set_field(counter, 0);
    }
    return fields;
  }
  if (uses_legacy_waitcnt(model.value()) && mnemonic == "s_waitcnt") {
    const auto value = operand_value(0);
    if (!value)
      return std::nullopt;
    const LegacyWaitcnt wait = decode_legacy_waitcnt(*value, arch);
    set_field(WaitCounterKind::Load, wait.vmcnt);
    set_field(WaitCounterKind::Exp, wait.expcnt);
    set_field(WaitCounterKind::Ds, wait.lgkmcnt);
    return fields;
  }

  if ((has_legacy_vscnt(model.value()) && mnemonic == "s_waitcnt_depctr") ||
      (!uses_legacy_waitcnt(model.value()) && mnemonic == "s_wait_alu")) {
    const auto value = operand_value(0);
    if (!value)
      return std::nullopt;
    set_field(WaitCounterKind::VaVdst, depctr_field(*value, 12, 4));
    set_field(WaitCounterKind::VmVsrc, depctr_field(*value, 2, 3));
    return fields;
  }

  if (uses_legacy_waitcnt(model.value())) {
    const bool sopk_wait = mnemonic == "s_waitcnt_vmcnt" || mnemonic == "s_waitcnt_vscnt" ||
                           mnemonic == "s_waitcnt_expcnt" || mnemonic == "s_waitcnt_lgkmcnt";
    if (!sopk_wait)
      return std::nullopt;
    const auto value = operand_value(1);
    if (!value)
      return std::nullopt;
    if (mnemonic == "s_waitcnt_vmcnt")
      set_field(WaitCounterKind::Load, *value);
    else if (mnemonic == "s_waitcnt_vscnt")
      set_field(WaitCounterKind::Store, *value);
    else if (mnemonic == "s_waitcnt_expcnt")
      set_field(WaitCounterKind::Exp, *value);
    else
      set_field(WaitCounterKind::Ds, *value);
    return fields;
  }

  const auto value = operand_value(0);
  if (!value)
    return std::nullopt;
  if (mnemonic == "s_wait_loadcnt")
    set_field(WaitCounterKind::Load, *value);
  else if (mnemonic == "s_wait_storecnt")
    set_field(WaitCounterKind::Store, *value);
  else if (mnemonic == "s_wait_dscnt")
    set_field(WaitCounterKind::Ds, *value);
  else if (mnemonic == "s_wait_kmcnt")
    set_field(WaitCounterKind::Km, *value);
  else if (mnemonic == "s_wait_samplecnt")
    set_field(WaitCounterKind::Sample, *value);
  else if (mnemonic == "s_wait_bvhcnt")
    set_field(WaitCounterKind::Bvh, *value);
  else if (mnemonic == "s_wait_expcnt")
    set_field(WaitCounterKind::Exp, *value);
  else if (mnemonic == "s_wait_xcnt")
    set_field(WaitCounterKind::X, *value);
  else if (mnemonic == "s_wait_asynccnt")
    set_field(WaitCounterKind::Async, *value);
  else if (mnemonic == "s_wait_tensorcnt")
    set_field(WaitCounterKind::Tensor, *value);
  else if (mnemonic == "s_wait_loadcnt_dscnt") {
    set_field(WaitCounterKind::Load, (*value >> 8u) & 0x3fu);
    set_field(WaitCounterKind::Ds, *value & 0x3fu);
  } else if (mnemonic == "s_wait_storecnt_dscnt") {
    set_field(WaitCounterKind::Store, (*value >> 8u) & 0x3fu);
    set_field(WaitCounterKind::Ds, *value & 0x3fu);
  } else {
    return std::nullopt;
  }
  return fields;
}

[[nodiscard]] util::FailureOr<std::optional<WaitFields>>
WaitcheckTarget::embedded_wait_fields(const Instruction &inst, rj_code_arch_t arch) {
  const auto model = waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  WaitFields fields;
  bool has_wait = false;
  if (const auto wait_exp = vinterp_wait_exp(inst); wait_exp && *wait_exp < 0x7u) {
    fields[counter_index(WaitCounterKind::Exp)] = *wait_exp;
    has_wait = true;
  }
  if (const auto wait_vm_vsrc = dsdir_wait_vm_vsrc(inst);
      !uses_legacy_waitcnt(model.value()) && wait_vm_vsrc && *wait_vm_vsrc == 0) {
    fields[counter_index(WaitCounterKind::VmVsrc)] = 0;
    has_wait = true;
  }
  if (const auto wait_va_vdst = dsdir_wait_va_vdst(inst); wait_va_vdst && *wait_va_vdst < 0xfu) {
    fields[counter_index(WaitCounterKind::VaVdst)] = *wait_va_vdst;
    has_wait = true;
  }
  return has_wait ? std::optional{fields} : std::nullopt;
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

[[nodiscard]] uint32_t WaitcheckTarget::depctr_field(uint32_t value, uint32_t shift,
                                                     uint32_t width) {
  return (value >> shift) & ((1u << width) - 1u);
}

[[nodiscard]] std::optional<int64_t> WaitcheckTarget::first_barrier_id(const Instruction &inst) {
  const Operand *op = inst.src_operand(0);
  if (!op)
    return std::nullopt;
  if (const auto value = op->const_value())
    return static_cast<int64_t>(*value);
  return static_cast<int64_t>(static_cast<int32_t>(op->encoding_value()));
}

[[nodiscard]] bool WaitcheckTarget::is_cdna4_mubuf_lds_load(const Instruction &inst,
                                                            rj_code_arch_t arch) {
  if ((arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4) ||
      !inst.mnemonic().starts_with("buffer_load") || inst.raw_encoding() == nullptr ||
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
         mnemonic.starts_with("s_prefetch_") || mnemonic.starts_with("s_atc_probe") ||
         mnemonic.starts_with("s_atomic_") || mnemonic.starts_with("s_buffer_atomic_") ||
         mnemonic.starts_with("s_scratch_load") || mnemonic.starts_with("s_scratch_store") ||
         mnemonic.starts_with("s_buffer_prefetch_") || mnemonic.starts_with("s_dcache_") ||
         mnemonic == "s_gl1_inv" || mnemonic == "s_memtime" || mnemonic == "s_memrealtime" ||
         mnemonic == "s_get_barrier_state";
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

[[nodiscard]] util::FailureOr<std::vector<ClassifiedEvent>>
WaitcheckTarget::classify_events(const Instruction &inst, rj_code_arch_t arch) {
  const auto model = waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  auto has_register_result = [&] {
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      if (const Operand *operand = inst.dst_operand(i); operand && operand->to_register_ref())
        return true;
    }
    return false;
  };
  std::vector<ClassifiedEvent> events;
  const bool expert = supports_expert_scheduling(arch);
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
    // These operations update both their ordinary VMEM counter and the
    // independent async counter in LLVM's hardware-event model.
    events.emplace_back(is_load ? WaitCounterKind::Load : vmem_store_wait_counter(model.value()),
                        is_load ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore,
                        TrackedRegisterSource::None, /*check_uses=*/false,
                        /*check_defs=*/false);
    events.emplace_back(WaitCounterKind::Async, kind, TrackedRegisterSource::None,
                        /*check_uses=*/false, /*check_defs=*/false,
                        /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/true, /*check_program_end=*/false);
    // LLVM excludes ASYNC_CNT/TENSOR_CNT from blanket function-boundary
    // waits. Their waits are derived from object-invisible ASYNCMARK pairs,
    // so conservatively check observable LDS consumers but not program end.
    add_xcnt_event(is_load ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore);
    return events;
  }

  if (mnemonic == "ds_atomic_async_barrier_arrive_b64") {
    events.emplace_back(WaitCounterKind::Async, WaitEventKind::AsyncBarrier,
                        TrackedRegisterSource::None, /*check_uses=*/false,
                        /*check_defs=*/false, /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/true,
                        /*check_program_end=*/false);
    return events;
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
    return events;
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
    return events;
  }

  if (mnemonic == "global_inv") {
    // LLVM tracks this in LOAD_CNT so it changes the wait threshold for an
    // older load.  It has no result and does not by itself make the next
    // memory instruction a wait consumer.
    events.emplace_back(WaitCounterKind::Load, WaitEventKind::GlobalInv,
                        TrackedRegisterSource::None, false, false);
    return events;
  }

  if (mnemonic == "global_wb" || mnemonic == "global_wbinv") {
    events.emplace_back(vmem_store_wait_counter(model.value()), WaitEventKind::GlobalWb,
                        TrackedRegisterSource::None, false, false);
    return events;
  }

  if (is_cdna4_mubuf_lds_load(inst, arch)) {
    events.emplace_back(WaitCounterKind::Load, WaitEventKind::LdsDirect,
                        TrackedRegisterSource::None, /*check_uses=*/false,
                        /*check_defs=*/false, /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/false,
                        /*check_program_end=*/false,
                        /*check_counter_parity_order=*/true);
    return events;
  }

  if (mnemonic.starts_with("global_load") || mnemonic.starts_with("scratch_load") ||
      mnemonic.starts_with("buffer_load") || mnemonic.starts_with("tbuffer_load") ||
      mnemonic.starts_with("image_load")) {
    events.push_back({WaitCounterKind::Load, WaitEventKind::VmemNoSamplerLoad});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemNoSamplerLoad,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::VmemNoSamplerLoad);
    return events;
  }

  if (is_image_atomic(mnemonic)) {
    // Image models expose vdata as a destination even for no-return forms.
    // Read the return control until that distinction is available as metadata:
    // gfx11 GLC is bit 14; gfx12 TH's low bit is bit 20 of the second word.
    bool returns_value = has_register_result();
    if (inst.raw_encoding() != nullptr && inst.size() >= 8) {
      returns_value = uses_legacy_waitcnt(model.value())
                          ? (inst.raw_encoding()[0] & (1u << 14u)) != 0
                          : (inst.raw_encoding()[1] & (1u << 20u)) != 0;
    }
    const WaitCounterKind counter =
        returns_value ? WaitCounterKind::Load : vmem_store_wait_counter(model.value());
    const WaitEventKind kind =
        returns_value ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore;
    events.emplace_back(counter, kind,
                        returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        /*check_uses=*/returns_value, /*check_defs=*/returns_value);
    if (!uses_legacy_waitcnt(model.value()))
      events.push_back({WaitCounterKind::Exp, WaitEventKind::VmemStore,
                        TrackedRegisterSource::StoreDataUses, false, true});
    if (expert)
      events.push_back(
          {WaitCounterKind::VmVsrc, kind, TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(kind);
    return events;
  }

  if (is_vmem_atomic(mnemonic)) {
    const bool returns_value = has_register_result();
    const WaitCounterKind counter =
        returns_value ? WaitCounterKind::Load : vmem_store_wait_counter(model.value());
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
    return events;
  }

  if (mnemonic.starts_with("flat_store")) {
    events.push_back({vmem_store_wait_counter(model.value()), WaitEventKind::FlatStore,
                      TrackedRegisterSource::None, false, false});
    events.push_back(
        {WaitCounterKind::Ds, WaitEventKind::FlatStore, TrackedRegisterSource::None, false, false});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::FlatStore,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::FlatStore);
    return events;
  }

  if (is_vmem_store(mnemonic)) {
    events.push_back({vmem_store_wait_counter(model.value()), WaitEventKind::VmemStore,
                      TrackedRegisterSource::None, false, false});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemStore,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::VmemStore);
    return events;
  }

  if (uses_ds_wait_counter(mnemonic)) {
    const uint32_t gds_bit =
        arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 16u : 17u;
    const bool gds = uses_legacy_waitcnt(model.value()) &&
                     (mnemonic == "ds_ordered_count" || mnemonic == "ds_add_gs_reg_rtn" ||
                      mnemonic == "ds_sub_gs_reg_rtn" || mnemonic.starts_with("ds_gws_") ||
                      (inst.raw_encoding() != nullptr && inst.size() >= 8 &&
                       (inst.raw_encoding()[0] & (1u << gds_bit)) != 0));
    events.push_back({WaitCounterKind::Ds, gds ? WaitEventKind::Gds : WaitEventKind::Ds});
    if (gds)
      events.push_back({WaitCounterKind::Exp, WaitEventKind::Gds, TrackedRegisterSource::VectorUses,
                        false, true, true});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Ds,
                        TrackedRegisterSource::VectorUses, false, true});
    return events;
  }

  if (mnemonic == "ds_param_load" || mnemonic == "ds_direct_load" || mnemonic == "lds_param_load" ||
      mnemonic == "lds_direct_load") {
    events.push_back({WaitCounterKind::Exp, WaitEventKind::LdsDirect});
    return events;
  }

  if (is_scalar_memory_op(mnemonic)) {
    const bool produces_result = has_register_result();
    events.emplace_back(smem_wait_counter(model.value()), WaitEventKind::Smem,
                        produces_result ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        /*check_uses=*/produces_result, /*check_defs=*/produces_result);
    // Timestamp and barrier-state SOP operations account as SMEM_ACCESS,
    // but unlike SMRD instructions they do not translate a scalar address.
    if (mnemonic != "s_memtime" && mnemonic != "s_memrealtime" && mnemonic != "s_get_barrier_state")
      add_xcnt_event(WaitEventKind::Smem);
    return events;
  }

  if (mnemonic == "s_sendmsg" || mnemonic == "s_sendmsghalt" || mnemonic == "s_sendmsg_rtn_b32" ||
      mnemonic == "s_sendmsg_rtn_b64") {
    const bool returns_value = has_register_result();
    events.emplace_back(smem_wait_counter(model.value()), WaitEventKind::SqMessage,
                        returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        returns_value, returns_value);
    return events;
  }

  if (mnemonic == "image_msaa_load" || mnemonic.starts_with("image_sample") ||
      mnemonic.starts_with("image_gather")) {
    events.push_back({image_sample_wait_counter(model.value()), WaitEventKind::Sample});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Sample,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::Sample);
    return events;
  }

  if (mnemonic.starts_with("image_bvh")) {
    events.push_back({image_bvh_wait_counter(model.value()), WaitEventKind::Bvh});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Bvh,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::Bvh);
    return events;
  }

  if (mnemonic == "export" || mnemonic == "exp") {
    events.push_back({WaitCounterKind::Exp, WaitEventKind::Export, TrackedRegisterSource::Uses,
                      false, true, true});
    return events;
  }

  if (mnemonic == "s_barrier_signal_isfirst") {
    if (!uses_legacy_waitcnt(model.value()))
      events.push_back({WaitCounterKind::Km, WaitEventKind::SccWrite, TrackedRegisterSource::None,
                        true, true, false, RegisterRef{RegClass::SCC, 0, 1},
                        first_barrier_id(inst)});
    return events;
  }

  return events;
}

} // namespace waitcheck_detail
} // namespace rocjitsu
