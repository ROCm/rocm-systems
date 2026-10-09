// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"

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
    if (no_wait && value < *no_wait)
      fields[counter_index(counter)] = value;
  };

  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic == "s_wait_idle" || (arch == ROCJITSU_CODE_ARCH_RDNA4 && mnemonic == "s_waitcnt")) {
    // RDNA4's compatibility S_WAITCNT is S_WAIT_IDLE regardless of SIMM16.
    for (size_t counter_idx = 0; counter_idx < kWaitCounterCount; ++counter_idx) {
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

[[nodiscard]] uint32_t WaitCounterPolicy::depctr_field(uint32_t value, uint32_t shift,
                                                       uint32_t width) {
  return (value >> shift) & ((1u << width) - 1u);
}

} // namespace rocjitsu
