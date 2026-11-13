// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

namespace rocjitsu::amdgpu {
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

[[nodiscard]] size_t WaitCounterPolicy::counter_index(WaitCounterKind counter) {
  return static_cast<size_t>(counter);
}

[[nodiscard]] util::FailureOr<uint32_t>
WaitCounterPolicy::maximum_dependency_wait(rj_code_arch_t arch, WaitCounterKind counter) {
  const auto model = waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  return maximum_dependency_wait(model.value(), counter);
}

[[nodiscard]] util::FailureOr<std::optional<uint32_t>>
WaitCounterPolicy::counter_no_wait_value(rj_code_arch_t arch, WaitCounterKind counter) {
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
      return has_legacy_vscnt(model.value()) ? 0x3fu : 0x0fu;
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
WaitCounterPolicy::explicit_wait_fields(const Instruction &inst, rj_code_arch_t arch) {
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
    if (no_wait) {
      // Only the counter field participates; upper SIMM16 bits are ignored.
      value &= *no_wait;
      if (value < *no_wait)
        fields[counter_index(counter)] = value;
    }
  };

  const Operation operation = inst.amdgpu_operation();
  if (operation == Operation::S_WAIT_IDLE ||
      (arch == ROCJITSU_CODE_ARCH_RDNA4 && operation == Operation::S_WAITCNT)) {
    // RDNA4's compatibility S_WAITCNT is S_WAIT_IDLE regardless of SIMM16.
    for (size_t counter_idx = 0; counter_idx < kWaitCounterCount; ++counter_idx) {
      const auto counter = static_cast<WaitCounterKind>(counter_idx);
      if (counter_no_wait_value(arch, counter).value())
        set_field(counter, 0);
    }
    return fields;
  }
  if (uses_legacy_waitcnt(model.value()) && operation == Operation::S_WAITCNT) {
    const auto value = operand_value(0);
    if (!value)
      return std::nullopt;
    const LegacyWaitcnt wait = decode_legacy_waitcnt(*value, arch);
    set_field(WaitCounterKind::Load, wait.vmcnt);
    set_field(WaitCounterKind::Exp, wait.expcnt);
    set_field(WaitCounterKind::Ds, wait.lgkmcnt);
    return fields;
  }

  if ((has_legacy_vscnt(model.value()) && operation == Operation::S_WAITCNT_DEPCTR) ||
      (!uses_legacy_waitcnt(model.value()) && operation == Operation::S_WAIT_ALU)) {
    const auto value = operand_value(0);
    if (!value)
      return std::nullopt;
    set_field(WaitCounterKind::VaVdst, depctr_field(*value, 12, 4));
    set_field(WaitCounterKind::VmVsrc, depctr_field(*value, 2, 3));
    return fields;
  }

  if (uses_legacy_waitcnt(model.value())) {
    const bool sopk_wait =
        operation == Operation::S_WAITCNT_VMCNT || operation == Operation::S_WAITCNT_VSCNT ||
        operation == Operation::S_WAITCNT_EXPCNT || operation == Operation::S_WAITCNT_LGKMCNT;
    if (!sopk_wait)
      return std::nullopt;
    const auto value = operand_value(1);
    if (!value)
      return std::nullopt;
    if (operation == Operation::S_WAITCNT_VMCNT)
      set_field(WaitCounterKind::Load, *value);
    else if (operation == Operation::S_WAITCNT_VSCNT)
      set_field(WaitCounterKind::Store, *value);
    else if (operation == Operation::S_WAITCNT_EXPCNT)
      set_field(WaitCounterKind::Exp, *value);
    else
      set_field(WaitCounterKind::Ds, *value);
    return fields;
  }

  const auto value = operand_value(0);
  if (!value)
    return std::nullopt;
  if (operation == Operation::S_WAIT_LOADCNT)
    set_field(WaitCounterKind::Load, *value);
  else if (operation == Operation::S_WAIT_STORECNT)
    set_field(WaitCounterKind::Store, *value);
  else if (operation == Operation::S_WAIT_DSCNT)
    set_field(WaitCounterKind::Ds, *value);
  else if (operation == Operation::S_WAIT_KMCNT)
    set_field(WaitCounterKind::Km, *value);
  else if (operation == Operation::S_WAIT_SAMPLECNT)
    set_field(WaitCounterKind::Sample, *value);
  else if (operation == Operation::S_WAIT_BVHCNT)
    set_field(WaitCounterKind::Bvh, *value);
  else if (operation == Operation::S_WAIT_EXPCNT)
    set_field(WaitCounterKind::Exp, *value);
  else if (operation == Operation::S_WAIT_XCNT)
    set_field(WaitCounterKind::X, *value);
  else if (operation == Operation::S_WAIT_ASYNCCNT)
    set_field(WaitCounterKind::Async, *value);
  else if (operation == Operation::S_WAIT_TENSORCNT)
    set_field(WaitCounterKind::Tensor, *value);
  else if (operation == Operation::S_WAIT_LOADCNT_DSCNT) {
    set_field(WaitCounterKind::Load, (*value >> 8u) & 0x3fu);
    set_field(WaitCounterKind::Ds, *value & 0x3fu);
  } else if (operation == Operation::S_WAIT_STORECNT_DSCNT) {
    set_field(WaitCounterKind::Store, (*value >> 8u) & 0x3fu);
    set_field(WaitCounterKind::Ds, *value & 0x3fu);
  } else {
    return std::nullopt;
  }
  return fields;
}

[[nodiscard]] uint32_t WaitCounterPolicy::depctr_field(uint32_t value, uint32_t shift,
                                                       uint32_t width) {
  return (value >> shift) & ((1u << width) - 1u);
}

} // namespace rocjitsu::amdgpu
