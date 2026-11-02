// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_inline_access.h
/// @brief DS instructions completing their LDS access inside execute_impl().
///
/// @details Reads top-down, one file per level:
///
///   lds_inline_access.h        the entry points generated handlers call
///   lds_inline_access_steps.h  their numbered steps: gate, addresses, access
///   lds_load_plans.h           loads: LDS -> staging, by address shape
///   lds_writeback.h            loads: staging -> VGPRs, by format and EXEC
///   lds_store_plans.h          stores: VGPR rows -> LDS, by EXEC and shape
///   lds_lane_shapes.h          the address-shape proofs loads and stores share
///
/// Each hot loop has a plain C++ body that builds anywhere and, where it
/// measured faster, an AVX2 body, which AVX2 builds use.
///
/// "The census" in these files is the DS accesses recorded across a corpus of
/// emulated workloads (BLAS, RNG, FFT, convolutions, Triton kernels). The
/// branch orders and the comments' "most" and "many" come from it.

#pragma once

#include "rocjitsu/vm/amdgpu/lds_store_plans.h"
#include "rocjitsu/vm/amdgpu/lds_writeback.h"

#include <cstdint>

namespace rocjitsu {
class Instruction;
namespace amdgpu {

class VectorMemState;
class Wavefront;

/// @brief A plain DS load or store, single- or two-address, as its generated
/// handler describes it: register numbers and shape, not register contents.
/// @details The specialized subset of VectorMemState such an access needs. It
/// has no per-lane address, payload or response buffers: the request path
/// reads its operands from, and writes its results to, the registers
/// themselves. Field order matches the designated initializers the generator
/// emits.
struct LdsAccessRequest {
  uint32_t addr_vgpr = 0;   ///< Physical address VGPR.
  uint32_t offset = 0;      ///< Byte offset, (offset1 << 8) | offset0.
  uint32_t offset2 = 0;     ///< Two-address forms: the second access's byte offset.
  uint32_t data_vgpr = 0;   ///< Stores: first payload VGPR.
  uint32_t data2_vgpr = 0;  ///< Two-address stores: the second access's first payload VGPR.
  uint32_t dst_vgpr = 0;    ///< Loads: first destination VGPR.
  uint8_t elem_size = 0;    ///< 1, 2, 4 or 8 bytes.
  uint8_t num_elems = 0;    ///< Elements per lane.
  bool sign_extend = false; ///< Loads of i8/i16.
  bool d16_hi = false;      ///< Loads write, stores read, the upper 16 bits.
  bool d16_lo = false;      ///< Loads write the lower 16 bits.
};

/// @brief One access of a plain load between its LDS read and its writeback:
/// where its data is, and which registers and lanes the writeback stores.
struct LdsStagedLoad {
  uint8_t *dst[4]; ///< Raw lanes of each destination VGPR.
  /// Lanes 16g..16g+15, lane-major: in staging, or in place in LDS when those
  /// lanes read consecutive bytes.
  const uint8_t *group[4];
  uint64_t exec;
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

  /// @brief Plain single-address store of format @p F, without a
  /// VectorMemState: the gate and the EXEC dispatch inlined into the opcode's
  /// handler, the writers behind them (lds_store_plans.h).
  template <lds_store::StoreFormat F>
  static bool store(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf);

  /// @brief Two-address load (ds_read2* / ds_load_2addr*) of format @p F: two
  /// loads from the same address lanes at @p req.offset and @p req.offset2,
  /// into @p req.dst_vgpr and the registers right after it.
  template <lds_writeback::DstFormat F>
  static bool load2(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf);

  /// @brief Two-address store (ds_write2* / ds_store_2addr*) of format @p F:
  /// @p req.data_vgpr at @p req.offset for every lane, then @p req.data2_vgpr
  /// at @p req.offset2, so the second wins where they overlap.
  template <lds_store::StoreFormat F>
  static bool store2(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf);

  /// @brief Every other DS form: run the routed path's access code on @p d.
  static bool complete(const Instruction &inst, VectorMemState &d, Wavefront &wf);

private:
  /// Steps 1-5 of a plain load, or of both loads of a two-address one, in
  /// lds_inline_access_steps.h. It stops before the writeback, with access a's
  /// data read through @p load[a].group (staging: @p staging + a * 64 *
  /// kMaxLaneBytes) and its destination in @p load[a].
  template <bool Dual>
  static bool read(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf,
                   uint8_t *staging, LdsStagedLoad *load);

  /// The plain or two-address store, in lds_inline_access_steps.h.
  template <lds_store::StoreFormat F, bool Dual>
  static bool write(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf);
};

} // namespace amdgpu
} // namespace rocjitsu

// load<F>() is inlined into each generated handler.
#include "rocjitsu/vm/amdgpu/lds_inline_access_steps.h"
