// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file compute_unit_isa.h
/// @brief Selecting a compute unit's ISA from its configured architecture.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/result.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace rocjitsu::amdgpu {

/// @brief Call `fn.template operator()<Isa>()` with the ISA of architecture @p arch.
/// @param fn Returns util::Result.
/// @returns @p fn's result, or a diagnosed failure for an architecture with no compute unit.
template <typename F>
util::Result with_compute_unit_isa(rj_code_arch_t arch, F &&fn,
                                   util::DiagnosticEmitter emit_error) {
  switch (arch) {
    // \NPI new ISA family: add its architecture-to-ISA mapping here.
  case ROCJITSU_CODE_ARCH_CDNA1:
    return fn.template operator()<cdna1::Isa>();
  case ROCJITSU_CODE_ARCH_CDNA2:
    return fn.template operator()<cdna2::Isa>();
  case ROCJITSU_CODE_ARCH_CDNA3:
    return fn.template operator()<cdna3::Isa>();
  case ROCJITSU_CODE_ARCH_CDNA4:
    return fn.template operator()<cdna4::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA1:
    return fn.template operator()<rdna1::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA2:
    return fn.template operator()<rdna2::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA3:
    return fn.template operator()<rdna3::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return fn.template operator()<rdna3_5::Isa>();
  case ROCJITSU_CODE_ARCH_RDNA4:
    return fn.template operator()<rdna4::Isa>();
  case ROCJITSU_CODE_ARCH_CDNA5:
    return fn.template operator()<cdna5::Isa>();
  default:
    return emit_error.emit() << "Unsupported architecture for ComputeUnit";
  }
}

/// @brief Create a compute unit whose execution shell is @p BaseComputeUnit.
/// @details Applies ComputeUnitCore::validate(), then completes @p BaseComputeUnit
/// with the ISA of @p config's architecture, as
/// IsaExecComputeUnit<BaseComputeUnit::kExecMode, Isa, BaseComputeUnit>, so a
/// model's CU class is a compute unit of any architecture rather than a wrapper
/// around one.
/// @tparam BaseComputeUnit ExecComputeUnit<Mode>, or a class derived from it
///                         whose constructor takes @p base_args before
///                         ExecComputeUnit's arguments.
/// @param name Human-readable name (e.g., "cu0").
/// @param config CU configuration parameters.
/// @param memory Shared GPU memory (not owned).
/// @param l2 Shared L2 cache (not owned).
/// @param emit_error Destination for failure diagnostics.
/// @param base_args Arguments for @p BaseComputeUnit's own state.
/// @returns The compute unit, or null after diagnosing an invalid configuration.
template <typename BaseComputeUnit, typename... BaseArgs>
std::unique_ptr<BaseComputeUnit>
create_compute_unit(std::string name, const ComputeUnitCore::Config &config, GpuMemory *memory,
                    L2Cache *l2, util::DiagnosticEmitter emit_error, BaseArgs &&...base_args) {
  static_assert(std::is_base_of_v<ExecComputeUnit<BaseComputeUnit::kExecMode>, BaseComputeUnit>,
                "a compute unit's shell derives from ExecComputeUnit");
  if (ComputeUnitCore::validate(config, emit_error).failed())
    return nullptr;
  std::unique_ptr<BaseComputeUnit> compute_unit;
  const util::Result result = with_compute_unit_isa(
      config.arch,
      [&]<GpuIsa Isa>() -> util::Result {
        compute_unit =
            std::make_unique<IsaExecComputeUnit<BaseComputeUnit::kExecMode, Isa, BaseComputeUnit>>(
                std::move(name), config, memory, l2, std::forward<BaseArgs>(base_args)...);
        return util::Result::success();
      },
      emit_error);
  return result.failed() ? nullptr : std::move(compute_unit);
}

} // namespace rocjitsu::amdgpu
