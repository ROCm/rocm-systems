// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_inline_access.h
/// @brief DS instructions completing their LDS access inside execute_impl().

#pragma once

#include <cstdint>

namespace rocjitsu {
class Instruction;
namespace amdgpu {

class VectorMemState;
class Wavefront;

/// @brief A plain single-address DS load or store, as its generated handler
/// describes it: register numbers and shape, not register contents.
/// @details Field order matches the designated initializers the generator emits.
struct LdsAccessRequest {
  uint32_t addr_vgpr = 0; ///< Physical address VGPR.
  uint32_t offset = 0;    ///< Byte offset, (offset1 << 8) | offset0.
  uint32_t data_vgpr = 0; ///< Stores: first payload VGPR.
  uint32_t dst_vgpr = 0;  ///< Loads: first destination VGPR.
  uint8_t elem_size = 0;  ///< 1, 2, 4 or 8 bytes.
  uint8_t num_elems = 0;  ///< Elements per lane.
  bool is_load = false;
  bool sign_extend = false; ///< Loads of i8/i16.
  bool d16_hi = false;      ///< Loads write, stores read, the upper 16 bits.
  bool d16_lo = false;      ///< Loads write the lower 16 bits.
};

/// @brief Completes DS accesses inside execute_impl() when the CU allows it,
/// instead of routing them to the local memory pipeline.
/// @details Both entry points return false, with no side effects, when the CU
/// keeps the routed order; the handler then sets its state on the instruction.
class LdsInlineAccess {
public:
  /// @brief Plain single-address load or store, without a VectorMemState.
  static bool complete(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf);

  /// @brief Every other DS form: run the routed path's access code on @p d.
  static bool complete(const Instruction &inst, VectorMemState &d, Wavefront &wf);
};

} // namespace amdgpu
} // namespace rocjitsu
