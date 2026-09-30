// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file exec_mode.h
/// @brief Execution mode enum and ExecBase alias for compile-time mode selection.

#ifndef SIMDOJO_SIM_EXEC_MODE_H_
#define SIMDOJO_SIM_EXEC_MODE_H_

#include <cstdint>
#include <type_traits>

namespace simdojo {

template <typename Base> class Functional;
template <typename Base> class Clocked;

/// @brief Execution mode for simulation components.
///
/// @details Selects between functional (untimed, batched) and clocked
/// (cycle-accurate, event-per-edge) execution. Used as a compile-time
/// template parameter on model components and links.
enum class ExecMode : uint8_t {
  FUNCTIONAL, ///< Untimed execution: one event per activation, batched work.
  CLOCKED,    ///< Cycle-accurate execution: one event per clock edge.
};

/// @brief Maps an ExecMode to the corresponding CRTP mixin base class.
///
/// @details Use this alias to write components that work in either mode:
/// @code
///   template <ExecMode Mode>
///   class MyComponent : public ExecBase<Mode, Component> { ... };
/// @endcode
///
/// @tparam Mode The execution mode (FUNCTIONAL or CLOCKED).
/// @tparam Base A Component-derived type (Component or CompositeComponent).
template <ExecMode Mode, typename Base>
using ExecBase = std::conditional_t<Mode == ExecMode::FUNCTIONAL, Functional<Base>, Clocked<Base>>;

} // namespace simdojo

#endif // SIMDOJO_SIM_EXEC_MODE_H_
