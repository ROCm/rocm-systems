// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file isa_traits.h
/// @brief Compile-time ISA trait mapping and GPU ISA concept.

#ifndef ROCJITSU_ISA_ISA_TRAITS_H_
#define ROCJITSU_ISA_ISA_TRAITS_H_

#include "rocjitsu/base/api.h"

#include <concepts>
#include <cstdint>

namespace rocjitsu {

/// @brief Compile-time mapping from rj_code_arch_t enum values to Isa trait types.
///
/// @details Specialize this template in each architecture's isa.h to bind its
/// namespace::Isa type. Unspecialized archs will cause a compile error if
/// used, catching unsupported archs early.
/// @tparam Arch Architecture enum value to map.
template <rj_code_arch_t Arch> struct IsaTrait;

/// @brief Concept for ISAs that define GPU wavefront properties.
///
/// Requires the static constants (WF_SIZE, MAX_SGPRS_PER_WF, MAX_VGPRS_PER_WF)
/// and the type aliases (Context, OperandType, StatusReg) that the Isa* class
/// templates depend on.
template <typename Isa>
concept GpuIsa = requires {
  { Isa::WF_SIZE } -> std::convertible_to<uint32_t>;
  { Isa::MAX_SGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::MAX_VGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  typename Isa::Context;
  typename Isa::OperandType;
  typename Isa::StatusReg;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_ISA_TRAITS_H_
