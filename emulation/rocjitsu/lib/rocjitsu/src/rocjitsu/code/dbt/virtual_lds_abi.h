// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file virtual_lds_abi.h
/// @brief Shared compile-time/runtime ABI for virtual-LDS dispatch state.

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocjitsu {

/// @brief GPU-visible state written by the HSA hook and read by an entry prologue.
struct VirtualLdsDispatchState {
  /// @brief GPU virtual address of the per-dispatch backing allocation.
  uint64_t backing_base = 0;
  /// @brief Bytes between adjacent workgroups in the X dimension.
  uint32_t stride_x = 0;
  /// @brief Bytes between adjacent workgroups in the Y dimension.
  uint32_t stride_y = 0;
  /// @brief Bytes between adjacent workgroups in the Z dimension.
  uint32_t stride_z = 0;
  /// @brief Reserved for a future ABI revision; writers must initialize it to zero.
  uint32_t reserved = 0;
};

/// @brief Byte offsets consumed by target entry-prologue load instructions.
///
/// @details Deriving these from the shared standard-layout structure prevents
/// the HSA hook's runtime writer and DBT's instruction emitter from maintaining
/// independent numeric copies of the same ABI.
inline constexpr uint32_t kVirtualLdsStateBackingBaseOffset =
    offsetof(VirtualLdsDispatchState, backing_base);
inline constexpr uint32_t kVirtualLdsStateStrideXOffset =
    offsetof(VirtualLdsDispatchState, stride_x);
inline constexpr uint32_t kVirtualLdsStateStrideYOffset =
    offsetof(VirtualLdsDispatchState, stride_y);
inline constexpr uint32_t kVirtualLdsStateStrideZOffset =
    offsetof(VirtualLdsDispatchState, stride_z);
/// @brief Total wrapper-extension bytes reserved for virtual-LDS runtime state.
inline constexpr uint32_t kVirtualLdsRuntimeStateBytes = sizeof(VirtualLdsDispatchState);

static_assert(std::is_standard_layout_v<VirtualLdsDispatchState>);
static_assert(sizeof(VirtualLdsDispatchState) == 24);

} // namespace rocjitsu
