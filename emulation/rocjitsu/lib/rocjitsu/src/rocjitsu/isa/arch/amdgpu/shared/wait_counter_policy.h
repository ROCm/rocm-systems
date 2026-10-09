// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter.h"
#include "util/result.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace rocjitsu {
class Instruction;

/// @brief Architectural wait-counter domains.
///
/// @details GFX12 targets use the split counter names directly. Legacy gfx9
/// style targets reuse Load for vmcnt, Ds for lgkmcnt, and Exp for expcnt.
enum class WaitCounterKind : uint8_t {
  Load = 0,
  Store,
  Ds,
  Km,
  Sample,
  Bvh,
  Exp,
  X,
  Async,
  Tensor,
  VmVsrc,
  VaVdst,
  Depctr,
  Count,
};

[[nodiscard]] std::string_view wait_counter_name(WaitCounterKind counter);

inline constexpr size_t kWaitCounterCount = static_cast<size_t>(WaitCounterKind::Count);
using WaitFields = std::array<std::optional<uint32_t>, kWaitCounterCount>;

enum class WaitcntModel { LegacyNoVscnt, LegacyVscnt, SplitGfx12 };

// Reject unsupported architectures before selecting any target policies.
[[nodiscard]] inline util::FailureOr<WaitcntModel> waitcnt_model(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return WaitcntModel::LegacyNoVscnt;
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return WaitcntModel::LegacyVscnt;
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_CDNA5:
    return WaitcntModel::SplitGfx12;
  default:
    return util::Result::failure();
  }
}

[[nodiscard]] inline bool uses_legacy_waitcnt(WaitcntModel model) {
  return model != WaitcntModel::SplitGfx12;
}

[[nodiscard]] inline bool has_legacy_vscnt(WaitcntModel model) {
  return model == WaitcntModel::LegacyVscnt;
}

[[nodiscard]] inline bool supports_expert_scheduling(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

// Counter retirement order does not imply ordered VGPR writeback on GFX12.
[[nodiscard]] inline bool has_ordered_vmem_writeback(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
}

[[nodiscard]] inline WaitCounterKind smem_wait_counter(WaitcntModel model) {
  return uses_legacy_waitcnt(model) ? WaitCounterKind::Ds : WaitCounterKind::Km;
}

[[nodiscard]] inline WaitCounterKind vmem_store_wait_counter(WaitcntModel model) {
  return model == WaitcntModel::LegacyNoVscnt ? WaitCounterKind::Load : WaitCounterKind::Store;
}

[[nodiscard]] inline WaitCounterKind image_sample_wait_counter(WaitcntModel model) {
  return uses_legacy_waitcnt(model) ? WaitCounterKind::Load : WaitCounterKind::Sample;
}

[[nodiscard]] inline WaitCounterKind image_bvh_wait_counter(WaitcntModel model) {
  return uses_legacy_waitcnt(model) ? WaitCounterKind::Load : WaitCounterKind::Bvh;
}

struct LegacyWaitcnt {
  uint32_t vmcnt = 0;
  uint32_t expcnt = 0;
  uint32_t lgkmcnt = 0;
};

[[nodiscard]] inline LegacyWaitcnt decode_legacy_waitcnt(uint32_t value) {
  return {
      (value & 0xFu) | (((value >> 14u) & 0x3u) << 4u),
      (value >> 4u) & 0x7u,
      (value >> 8u) & 0xFu,
  };
}

[[nodiscard]] inline LegacyWaitcnt decode_gfx11_waitcnt(uint32_t value) {
  return {
      (value >> 10u) & 0x3Fu,
      value & 0x7u,
      (value >> 4u) & 0x3Fu,
  };
}

[[nodiscard]] inline LegacyWaitcnt decode_legacy_waitcnt(uint32_t value, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5)
    return decode_gfx11_waitcnt(value);
  auto fields = decode_legacy_waitcnt(value);
  // GFX10 retains the GFX9 positions but widens LGKM to six bits.
  if (arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2)
    fields.lgkmcnt = (value >> 8u) & 0x3fu;
  return fields;
}

[[nodiscard]] inline util::FailureOr<std::string>
wait_expression(WaitCounterKind counter, uint32_t required_count, rj_code_arch_t arch) {
  const auto model = waitcnt_model(arch);
  if (model.failed())
    return util::Result::failure();
  std::ostringstream os;
  if (uses_legacy_waitcnt(model.value())) {
    switch (counter) {
    case WaitCounterKind::Load:
      os << "s_waitcnt vmcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Ds:
      os << "s_waitcnt lgkmcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Exp:
      os << "s_waitcnt expcnt(" << required_count << ")";
      return os.str();
    case WaitCounterKind::Store:
      if (has_legacy_vscnt(model.value())) {
        os << "s_waitcnt_vscnt null, " << required_count;
        return os.str();
      }
      break;
    default:
      break;
    }
  }
  if (counter == WaitCounterKind::X) {
    os << "s_wait_xcnt " << required_count;
  } else if (counter == WaitCounterKind::VmVsrc || counter == WaitCounterKind::VaVdst) {
    os << (has_legacy_vscnt(model.value()) ? "s_waitcnt_depctr " : "s_wait_alu ")
       << (counter == WaitCounterKind::VmVsrc ? "depctr_vm_vsrc(" : "depctr_va_vdst(")
       << required_count << ")";
  } else {
    os << "s_wait_" << wait_counter_name(counter) << " <= " << required_count;
  }
  return os.str();
}

/// Map a runtime memory-pipeline counter to its architectural diagnostic domain.
/// The runtime retains split event subsets even on targets with combined waits.
[[nodiscard]] constexpr WaitCounterKind canonical_wait_counter(amdgpu::WaitCounterType counter,
                                                               WaitcntModel model) {
  using amdgpu::WaitCounterType;
  switch (counter) {
  case WaitCounterType::VMCNT:
  case WaitCounterType::LOADCNT:
    return WaitCounterKind::Load;
  case WaitCounterType::VSCNT:
  case WaitCounterType::STORECNT:
    return model == WaitcntModel::LegacyNoVscnt ? WaitCounterKind::Load : WaitCounterKind::Store;
  case WaitCounterType::LGKMCNT:
  case WaitCounterType::DSCNT:
    return WaitCounterKind::Ds;
  case WaitCounterType::KMCNT:
    return model == WaitcntModel::SplitGfx12 ? WaitCounterKind::Km : WaitCounterKind::Ds;
  case WaitCounterType::EXPCNT:
    return WaitCounterKind::Exp;
  case WaitCounterType::ASYNCCNT:
    return WaitCounterKind::Async;
  case WaitCounterType::TENSORCNT:
    return WaitCounterKind::Tensor;
  }
  return WaitCounterKind::Count;
}

/// Returns nullopt for domains without a corresponding memory-pipeline counter.
[[nodiscard]] constexpr std::optional<amdgpu::WaitCounterType>
memory_wait_counter_type(WaitCounterKind counter, WaitcntModel model) {
  using amdgpu::WaitCounterType;
  switch (counter) {
  case WaitCounterKind::Load:
    return model == WaitcntModel::SplitGfx12 ? WaitCounterType::LOADCNT : WaitCounterType::VMCNT;
  case WaitCounterKind::Store:
    if (model == WaitcntModel::LegacyNoVscnt)
      return std::nullopt;
    return model == WaitcntModel::SplitGfx12 ? WaitCounterType::STORECNT : WaitCounterType::VSCNT;
  case WaitCounterKind::Ds:
    return model == WaitcntModel::SplitGfx12 ? WaitCounterType::DSCNT : WaitCounterType::LGKMCNT;
  case WaitCounterKind::Km:
    if (model == WaitcntModel::SplitGfx12)
      return WaitCounterType::KMCNT;
    return std::nullopt;
  case WaitCounterKind::Exp:
    return WaitCounterType::EXPCNT;
  case WaitCounterKind::Async:
    if (model == WaitcntModel::SplitGfx12)
      return WaitCounterType::ASYNCCNT;
    return std::nullopt;
  case WaitCounterKind::Tensor:
    if (model == WaitcntModel::SplitGfx12)
      return WaitCounterType::TENSORCNT;
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

/// Architectural wait encodings and counter limits.
struct WaitCounterPolicy {
  [[nodiscard]] static size_t counter_index(WaitCounterKind counter);

  [[nodiscard]] static util::FailureOr<uint32_t> maximum_dependency_wait(rj_code_arch_t arch,
                                                                         WaitCounterKind counter);
  [[nodiscard]] static constexpr uint32_t maximum_dependency_wait(WaitcntModel model,
                                                                  WaitCounterKind counter) {
    // LLVM caps a dependency score at the largest non-sentinel wait value.
    // The all-ones encoding means "no wait", so the largest useful value is
    // one less than the hardware counter mask.
    switch (counter) {
    case WaitCounterKind::Load:
    case WaitCounterKind::Store:
      return 62;
    case WaitCounterKind::Ds:
      return model == WaitcntModel::LegacyNoVscnt ? 14 : 62;
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

  [[nodiscard]] static util::FailureOr<std::optional<uint32_t>>
  counter_no_wait_value(rj_code_arch_t arch, WaitCounterKind counter);

  // Absent fields leave their counters unchanged. An engaged result with no
  // active fields represents an explicit no-wait sentinel.
  [[nodiscard]] static util::FailureOr<std::optional<WaitFields>>
  explicit_wait_fields(const Instruction &inst, rj_code_arch_t arch);

  [[nodiscard]] static uint32_t depctr_field(uint32_t value, uint32_t shift, uint32_t width);
};

} // namespace rocjitsu
