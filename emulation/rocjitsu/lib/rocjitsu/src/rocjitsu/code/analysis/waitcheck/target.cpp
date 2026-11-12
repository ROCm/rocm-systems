// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/waitcheck/target.h"

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu {
namespace waitcheck_detail {
using amdgpu::MemoryAddressing;
using amdgpu::MemoryKind;
using amdgpu::Operation;

std::optional<RegisterRef> WaitcheckTarget::register_operand(const Operand &operand,
                                                             rj_code_arch_t arch) {
  if (const auto ref = operand.to_register_ref())
    return ref;
  // Mesh exports read M0 through a fieldless input. Scalar instructions such
  // as S_SENDMSG_RTN_B32 can produce the same register through a selector.
  if ((amdgpu::has_typed_export_waits(arch) && operand.to_special_reg_class() == RegClass::M0) ||
      (operand.has_register_selector() &&
       static_cast<uint32_t>(operand.encoding_value()) == amdgpu::scalar_m0_selector(arch)))
    return RegisterRef{RegClass::M0, 0, 1};
  return std::nullopt;
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
    // GLOBAL_INV is explicitly ignored by LLVM's LOAD_CNT out-of-order
    // test. Pre-gfx12 image event kinds share that same hardware event.
    if (kind == WaitEventKind::GlobalInv)
      return std::nullopt;
    if (kind == WaitEventKind::FlatLoad || kind == WaitEventKind::LdsDirect ||
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
    if (is_xcnt_vmem_kind(kind))
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

bool WaitcheckTarget::is_xcnt_drain(const Instruction &inst) {
  return inst.flags() &
         (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR | XCNT_DRAIN);
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
  const MemoryAddressing addressing = inst.amdgpu_memory_addressing();
  return inst.transfers_to_lds() && inst.amdgpu_memory_kind() == MemoryKind::Load &&
         (addressing == MemoryAddressing::Scratch || addressing == MemoryAddressing::Global ||
          addressing == MemoryAddressing::Buffer);
}

[[nodiscard]] std::optional<uint32_t> WaitcheckTarget::vinterp_wait_exp(const Instruction &inst) {
  if (!inst.is_interpolation() || inst.raw_encoding() == nullptr ||
      inst.size() < static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  return (inst.raw_encoding()[0] >> 8u) & 0x7u;
}

[[nodiscard]] bool WaitcheckTarget::is_dsdir(const Instruction &inst) {
  return inst.amdgpu_memory_addressing() == MemoryAddressing::DirectLds;
}

[[nodiscard]] std::optional<uint32_t> WaitcheckTarget::dsdir_wait_va_vdst(const Instruction &inst) {
  if (!is_dsdir(inst) || inst.raw_encoding() == nullptr ||
      inst.size() < static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  return (inst.raw_encoding()[0] >> 16u) & 0xFu;
}

[[nodiscard]] std::optional<uint32_t> WaitcheckTarget::dsdir_wait_vm_vsrc(const Instruction &inst) {
  const Operation operation = inst.amdgpu_operation();
  if ((operation != Operation::DS_PARAM_LOAD && operation != Operation::DS_DIRECT_LOAD) ||
      inst.raw_encoding() == nullptr || inst.size() < static_cast<int>(sizeof(uint32_t)))
    return std::nullopt;
  return (inst.raw_encoding()[0] >> 23u) & 0x1u;
}

[[nodiscard]] bool WaitcheckTarget::is_scalar_memory_op(const Instruction &inst) {
  switch (inst.amdgpu_operation()) {
  case Operation::S_MEMTIME:
  case Operation::S_MEMREALTIME:
  case Operation::S_GET_BARRIER_STATE:
  case Operation::S_GET_WAVEID_IN_WORKGROUP:
    return true;
  default:
    break;
  }
  if (!(inst.flags() & SMEM))
    return false;
  switch (inst.amdgpu_memory_kind()) {
  case MemoryKind::Load:
  case MemoryKind::Store:
  case MemoryKind::Atomic:
  case MemoryKind::Probe:
  case MemoryKind::Cache:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool WaitcheckTarget::is_vmem_store(const Instruction &inst) {
  if (inst.amdgpu_memory_kind() != MemoryKind::Store || (inst.flags() & SMEM))
    return false;
  switch (inst.amdgpu_memory_addressing()) {
  case MemoryAddressing::Global:
  case MemoryAddressing::Scratch:
  case MemoryAddressing::Buffer:
  case MemoryAddressing::TypedBuffer:
  case MemoryAddressing::Image:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool WaitcheckTarget::is_vmem_atomic(const Instruction &inst) {
  if (inst.amdgpu_memory_kind() != MemoryKind::Atomic)
    return false;
  switch (inst.amdgpu_memory_addressing()) {
  case MemoryAddressing::Global:
  case MemoryAddressing::Flat:
  case MemoryAddressing::Buffer:
  case MemoryAddressing::Image:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool WaitcheckTarget::is_image_atomic(const Instruction &inst) {
  return inst.amdgpu_memory_addressing() == MemoryAddressing::Image &&
         inst.amdgpu_memory_kind() == MemoryKind::Atomic;
}

[[nodiscard]] bool WaitcheckTarget::uses_ds_wait_counter(const Instruction &inst) {
  // All VDS instructions use DS_CNT except async-barrier arrive, which uses
  // ASYNC_CNT. LDS-direct instructions have separate EXP_CNT semantics.
  const MemoryAddressing addressing = inst.amdgpu_memory_addressing();
  return (addressing == MemoryAddressing::Lds || addressing == MemoryAddressing::Gds) &&
         inst.amdgpu_operation() != Operation::DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64;
}

[[nodiscard]] bool WaitcheckTarget::is_async_lds_load(const Instruction &inst) {
  switch (inst.amdgpu_operation()) {
  case Operation::GLOBAL_LOAD_ASYNC_TO_LDS_B8:
  case Operation::GLOBAL_LOAD_ASYNC_TO_LDS_B32:
  case Operation::GLOBAL_LOAD_ASYNC_TO_LDS_B64:
  case Operation::GLOBAL_LOAD_ASYNC_TO_LDS_B128:
  case Operation::CLUSTER_LOAD_ASYNC_TO_LDS_B8:
  case Operation::CLUSTER_LOAD_ASYNC_TO_LDS_B32:
  case Operation::CLUSTER_LOAD_ASYNC_TO_LDS_B64:
  case Operation::CLUSTER_LOAD_ASYNC_TO_LDS_B128:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool WaitcheckTarget::is_async_lds_store(const Instruction &inst) {
  switch (inst.amdgpu_operation()) {
  case Operation::GLOBAL_STORE_ASYNC_FROM_LDS_B8:
  case Operation::GLOBAL_STORE_ASYNC_FROM_LDS_B32:
  case Operation::GLOBAL_STORE_ASYNC_FROM_LDS_B64:
  case Operation::GLOBAL_STORE_ASYNC_FROM_LDS_B128:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool WaitcheckTarget::is_tensor_lds_load(const Instruction &inst) {
  return inst.amdgpu_operation() == Operation::TENSOR_LOAD_TO_LDS;
}

[[nodiscard]] bool WaitcheckTarget::is_tensor_lds_store(const Instruction &inst) {
  return inst.amdgpu_operation() == Operation::TENSOR_STORE_FROM_LDS;
}

[[nodiscard]] util::FailureOr<std::vector<ClassifiedEvent>>
WaitcheckTarget::classify_events(const Instruction &inst, rj_code_arch_t arch) {
  std::vector<ClassifiedEvent> events;
  if (classify_events_into(inst, arch, events).failed())
    return util::Result::failure();
  return events;
}

[[nodiscard]] util::Result
WaitcheckTarget::classify_events_into(const Instruction &inst, rj_code_arch_t arch,
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
  const Operation operation = inst.amdgpu_operation();
  const MemoryAddressing addressing = inst.amdgpu_memory_addressing();
  const MemoryKind memory_kind = inst.amdgpu_memory_kind();
  auto add_xcnt_event = [&](WaitEventKind kind) {
    if (arch == ROCJITSU_CODE_ARCH_CDNA5)
      events.emplace_back(WaitCounterKind::X, kind, TrackedRegisterSource::Uses,
                          /*check_uses=*/false, /*check_defs=*/true,
                          /*check_exec_defs=*/is_xcnt_vmem_kind(kind));
  };
  if (is_async_lds_load(inst) || is_async_lds_store(inst)) {
    const bool is_load = is_async_lds_load(inst);
    const WaitEventKind kind = is_load ? WaitEventKind::AsyncLdsLoad : WaitEventKind::AsyncLdsStore;
    // These operations update both their ordinary VMEM counter and the
    // independent async counter in LLVM's hardware-event model.
    events.emplace_back(is_load ? WaitCounterKind::Load
                                : amdgpu::vmem_store_wait_counter(model.value()),
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
    return util::Result::success();
  }

  if (operation == Operation::DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64) {
    events.emplace_back(WaitCounterKind::Async, WaitEventKind::AsyncBarrier,
                        TrackedRegisterSource::None, /*check_uses=*/false,
                        /*check_defs=*/false, /*check_exec_defs=*/false, std::nullopt, std::nullopt,
                        /*check_memory_order=*/true,
                        /*check_program_end=*/false);
    return util::Result::success();
  }

  if (uses_ds_wait_counter(inst)) {
    const bool gds =
        amdgpu::uses_legacy_waitcnt(model.value()) && addressing == MemoryAddressing::Gds;
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

  if (is_tensor_lds_load(inst) || is_tensor_lds_store(inst)) {
    const bool is_load = is_tensor_lds_load(inst);
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

  if (addressing == MemoryAddressing::Flat && memory_kind == MemoryKind::Load) {
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

  if (operation == Operation::GLOBAL_INV || operation == Operation::BUFFER_INV ||
      operation == Operation::BUFFER_WBL2) {
    // LLVM tracks this in LOAD_CNT so it changes the wait threshold for an
    // older load.  It has no result and does not by itself make the next
    // memory instruction a wait consumer.
    events.emplace_back(WaitCounterKind::Load, WaitEventKind::GlobalInv,
                        TrackedRegisterSource::None, false, false);
    return util::Result::success();
  }

  if (operation == Operation::GLOBAL_WB || operation == Operation::GLOBAL_WBINV) {
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

  if ((memory_kind == MemoryKind::Load && !(inst.flags() & SMEM) &&
       (addressing == MemoryAddressing::Global || addressing == MemoryAddressing::Scratch ||
        addressing == MemoryAddressing::Buffer || addressing == MemoryAddressing::TypedBuffer ||
        (addressing == MemoryAddressing::Image && operation != Operation::IMAGE_MSAA_LOAD))) ||
      operation == Operation::IMAGE_GET_RESINFO) {
    events.push_back({WaitCounterKind::Load, WaitEventKind::VmemNoSamplerLoad});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemNoSamplerLoad,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::VmemNoSamplerLoad);
    return util::Result::success();
  }

  if (is_image_atomic(inst)) {
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

  if (is_vmem_atomic(inst)) {
    const bool returns_value = has_register_result();
    const WaitCounterKind counter =
        returns_value ? WaitCounterKind::Load : amdgpu::vmem_store_wait_counter(model.value());
    const WaitEventKind kind =
        addressing == MemoryAddressing::Flat
            ? (returns_value ? WaitEventKind::FlatLoad : WaitEventKind::FlatStore)
            : (returns_value ? WaitEventKind::VmemNoSamplerLoad : WaitEventKind::VmemStore);
    events.emplace_back(counter, kind,
                        returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        /*check_uses=*/returns_value, /*check_defs=*/returns_value);
    if (addressing == MemoryAddressing::Flat)
      events.emplace_back(WaitCounterKind::Ds, kind,
                          returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                          /*check_uses=*/returns_value, /*check_defs=*/returns_value);
    if (expert)
      events.emplace_back(WaitCounterKind::VmVsrc, kind, TrackedRegisterSource::VectorUses,
                          /*check_uses=*/false, /*check_defs=*/true);
    add_xcnt_event(kind);
    return util::Result::success();
  }

  if (addressing == MemoryAddressing::Flat && memory_kind == MemoryKind::Store) {
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

  if (is_vmem_store(inst)) {
    events.push_back({amdgpu::vmem_store_wait_counter(model.value()), WaitEventKind::VmemStore,
                      TrackedRegisterSource::None, false, false});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::VmemStore,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::VmemStore);
    return util::Result::success();
  }

  if (is_dsdir(inst)) {
    events.push_back({WaitCounterKind::Exp, WaitEventKind::LdsDirect});
    return util::Result::success();
  }

  if (is_scalar_memory_op(inst)) {
    const bool produces_result = has_register_result();
    events.emplace_back(amdgpu::smem_wait_counter(model.value()), WaitEventKind::Smem,
                        produces_result ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        /*check_uses=*/produces_result, /*check_defs=*/produces_result);
    // Timestamp and barrier-state SOP operations account as SMEM_ACCESS,
    // but unlike SMRD instructions they do not translate a scalar address.
    if (operation != Operation::S_MEMTIME && operation != Operation::S_MEMREALTIME &&
        operation != Operation::S_GET_BARRIER_STATE &&
        operation != Operation::S_GET_WAVEID_IN_WORKGROUP)
      add_xcnt_event(WaitEventKind::Smem);
    return util::Result::success();
  }

  if (operation == Operation::S_SENDMSG || operation == Operation::S_SENDMSGHALT ||
      operation == Operation::S_SENDMSG_RTN_B32 || operation == Operation::S_SENDMSG_RTN_B64) {
    const bool returns_value = has_register_result();
    events.emplace_back(amdgpu::smem_wait_counter(model.value()), WaitEventKind::SqMessage,
                        returns_value ? TrackedRegisterSource::Defs : TrackedRegisterSource::None,
                        returns_value, returns_value);
    return util::Result::success();
  }

  if (operation == Operation::IMAGE_MSAA_LOAD || operation == Operation::IMAGE_GET_LOD ||
      (addressing == MemoryAddressing::Image && memory_kind == MemoryKind::Sample)) {
    events.push_back({amdgpu::image_sample_wait_counter(model.value()), WaitEventKind::Sample});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Sample,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::Sample);
    return util::Result::success();
  }

  if (addressing == MemoryAddressing::Image && memory_kind == MemoryKind::Bvh) {
    events.push_back({amdgpu::image_bvh_wait_counter(model.value()), WaitEventKind::Bvh});
    if (expert)
      events.push_back({WaitCounterKind::VmVsrc, WaitEventKind::Bvh,
                        TrackedRegisterSource::VectorUses, false, true});
    add_xcnt_event(WaitEventKind::Bvh);
    return util::Result::success();
  }

  if (operation == Operation::EXPORT || operation == Operation::EXP) {
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

  if (operation == Operation::S_BARRIER_SIGNAL_ISFIRST) {
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
