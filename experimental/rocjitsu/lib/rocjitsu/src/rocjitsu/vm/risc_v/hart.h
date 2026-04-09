// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_RISC_V_HART_H_
#define ROCJITSU_VM_RISC_V_HART_H_

#include "rocjitsu/vm/risc_v/hart_state.h"
#include "rocjitsu/vm/risc_v/memory.h"
#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/clocked.h"

#include <string>

namespace rocjitsu {
namespace risc_v {

/// @brief A RISC-V hart modeled as a clocked simulation component.
///
/// Single-stage pipeline: on each clock edge, fetches one instruction,
/// decodes it, executes it, and advances the PC.
class Hart : public simdojo::Clocked<simdojo::Component> {
public:
  /// @brief Construct a RISC-V hart.
  /// @param[in] name Human-readable component name.
  /// @param[in] domain Clock domain that provides period and phase.
  Hart(std::string name, const simdojo::ClockDomain &domain);

  /// @brief Execute one instruction.
  /// @param[in] now Current simulation tick.
  /// @returns true if the hart is still active.
  bool advance(simdojo::Tick now) override;

  /// @brief Shutdown the hart after simulation ends.
  void shutdown() override;

  /// @brief Access the hart register/memory state.
  /// @returns Reference to the hart state.
  HartState &state() { return state_; }

  /// @brief Access the hart register/memory state.
  /// @returns Const reference to the hart state.
  const HartState &state() const { return state_; }

  /// @brief Access the hart memory subsystem.
  /// @returns Reference to the memory interface.
  Memory &memory() { return memory_; }

  /// @brief Access the hart memory subsystem.
  /// @returns Const reference to the memory interface.
  const Memory &memory() const { return memory_; }

private:
  HartState state_;
  Memory memory_{"memory"};
};

} // namespace risc_v
} // namespace rocjitsu

#endif // ROCJITSU_VM_RISC_V_HART_H_
