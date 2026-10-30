// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_inline_access.h
/// @brief DS instructions completing their LDS access inside execute_impl().

#pragma once

#include "rocjitsu/vm/amdgpu/lds_writeback.h"

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
  uint32_t addr_vgpr = 0;   ///< Physical address VGPR.
  uint32_t offset = 0;      ///< Byte offset, (offset1 << 8) | offset0.
  uint32_t data_vgpr = 0;   ///< Stores: first payload VGPR.
  uint32_t dst_vgpr = 0;    ///< Loads: first destination VGPR.
  uint8_t elem_size = 0;    ///< 1, 2, 4 or 8 bytes.
  uint8_t num_elems = 0;    ///< Elements per lane.
  bool sign_extend = false; ///< Loads of i8/i16.
  bool d16_hi = false;      ///< Loads write, stores read, the upper 16 bits.
  bool d16_lo = false;      ///< Loads write the lower 16 bits.
};

/// @brief A plain load between its LDS read and its writeback: where its data
/// is, and where and how the writeback stores it.
struct LdsStagedLoad {
  uint8_t *dst[4]; ///< Raw lanes of each destination VGPR.
  /// Lanes 16g..16g+15, lane-major: in staging, or in place in LDS when those
  /// lanes read consecutive bytes.
  const uint8_t *group[4];
  uint64_t exec;
  uint32_t wave;
  bool sram_ecc;
};

/// @brief Completes DS accesses inside execute_impl() when the CU allows it,
/// instead of routing them to the local memory pipeline.
/// @details Every entry point returns false, with no side effects, when the CU
/// keeps the routed order; the handler then sets its state on the instruction.
class LdsInlineAccess {
public:
  /// Largest per-lane footprint of a plain DS access (b128).
  static constexpr uint32_t kMaxLaneBytes = 16;

  /// @brief Plain single-address load of format @p F, without a
  /// VectorMemState: the LDS read, then the writeback for @p F, both inlined
  /// into the opcode's handler (lds_inline_access_steps.h).
  template <lds_writeback::DstFormat F>
  static bool load(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf);

  /// @brief Plain single-address store, without a VectorMemState.
  static bool store(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf);

  /// @brief Every other DS form: run the routed path's access code on @p d.
  static bool complete(const Instruction &inst, VectorMemState &d, Wavefront &wf);

private:
  /// Steps 1-6 of a plain access, in lds_inline_access_steps.h: a store
  /// completes; a load stops before its writeback, with its data read through
  /// @p load.group and its destination in @p load.
  template <bool IsLoad>
  static bool access(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf,
                     uint8_t *staging, LdsStagedLoad &load);
};

} // namespace amdgpu
} // namespace rocjitsu

// load<F>() is inlined into each generated handler.
#include "rocjitsu/vm/amdgpu/lds_inline_access_steps.h"
