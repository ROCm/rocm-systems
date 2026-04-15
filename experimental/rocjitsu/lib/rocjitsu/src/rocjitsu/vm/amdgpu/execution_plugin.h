// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu {
class Instruction;
namespace amdgpu {
class Wavefront;

/// @brief Abstract plugin interface for execution hooks.
///
/// Plugins receive callbacks at key points during kernel dispatch and
/// execution. The CommandProcessor owns plugins and passes them to CUs
/// and wavefronts. All hooks have empty default implementations.
class ExecutionPlugin {
public:
  virtual ~ExecutionPlugin() = default;

  /// Called when a workgroup is about to be dispatched.
  virtual void onWorkgroupDispatch(uint32_t /*wg_id*/, uint32_t /*lds_size*/,
                                   uint32_t /*num_waves*/,
                                   uint32_t /*vgpr_count*/,
                                   uint32_t /*sgpr_count*/) {}

  /// Called when a wavefront is dispatched to a CU.
  virtual void onWavefrontDispatch(Wavefront * /*wf*/, uint32_t /*wg_id*/,
                                   uint32_t /*wave_index*/) {}

  /// Called when a memory instruction is routed to a pipeline.
  virtual void onMemoryInstruction(Instruction * /*inst*/, Wavefront & /*wf*/) {}

  /// Called when a VGPR is read during instruction execution.
  virtual void onVgprRead(Wavefront * /*wf*/, uint32_t /*logical_reg*/,
                          uint32_t /*lane*/) {}

  /// Called when an SGPR is read during instruction execution.
  virtual void onSgprRead(Wavefront * /*wf*/, uint32_t /*logical_reg*/) {}

  /// Called when s_waitcnt sets counter thresholds.
  virtual void onWaitcnt(Wavefront * /*wf*/, int /*vmcnt*/, int /*lgkmcnt*/) {}
};

} // namespace amdgpu
} // namespace rocjitsu
