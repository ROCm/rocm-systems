// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_MEM_STATE_H_
#define ROCJITSU_VM_AMDGPU_MEM_STATE_H_

/// @file Dynamic pipeline state for AMDGPU memory instructions.
///
/// These are plain data containers attached to instructions via the
/// DynamicInstState slot on the Instruction base class. The memory
/// pipeline subclasses own the initiate/complete logic that operates
/// on this state.

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <array>
#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief Pipeline routing tags for AMDGPU memory instructions.
enum MemPipelineTag : uint8_t {
  SCALAR_MEM = 1,
  GLOBAL_MEM = 2,
  LOCAL_MEM = 3,
};

/// @brief Atomic read-modify-write operation type.
enum class AtomicOp : uint8_t {
  NONE = 0, ///< Not an atomic operation.
  SWAP,     ///< Exchange.
  CMPSWAP,  ///< Compare-and-swap (data[0] = src, data[1] = cmp).
  ADD,      ///< Atomic add.
  SUB,      ///< Atomic subtract (mem - data).
  RSUB,     ///< Atomic reverse subtract (data - mem).
  SMIN,     ///< Signed minimum.
  UMIN,     ///< Unsigned minimum.
  SMAX,     ///< Signed maximum.
  UMAX,     ///< Unsigned maximum.
  AND,      ///< Bitwise AND.
  OR,       ///< Bitwise OR.
  XOR,      ///< Bitwise XOR.
  INC,      ///< Increment (wrapping).
  DEC,      ///< Decrement (wrapping).
  FADD,     ///< Floating-point add.
  FMIN,     ///< Floating-point minimum.
  FMAX,     ///< Floating-point maximum.
};

/// @brief Dynamic pipeline state for scalar memory instructions (SMEM).
struct ScalarMemState : DynamicInstState {
  ScalarMemState() { tag_ = SCALAR_MEM; }
  uint64_t addr = 0;
  uint32_t dst_reg_base = 0;
  uint32_t num_dwords = 0;
  bool is_load = true;
  Mtype mtype = Mtype::RW;
  uint32_t response_data[16] = {};
  uint32_t store_data[16] = {};
};

/// @brief Dynamic pipeline state for vector memory instructions
/// (FLAT, MUBUF, MTBUF, DS).
struct VectorMemState : DynamicInstState {
  VectorMemState(MemPipelineTag pipeline) { tag_ = pipeline; }
  std::array<uint64_t, 64> per_lane_addr = {};
  uint64_t lane_mask = 0;
  uint32_t wf_size = 64; ///< Wavefront width (set from wavefront's wf_size()).
  uint32_t dst_reg_base = 0;
  uint32_t elem_size = 0;
  uint32_t num_elems = 0;
  bool is_load = true;
  Mtype mtype = Mtype::RW;
  bool non_temporal = false;
  bool sign_extend = false;
  AtomicOp atomic_op = AtomicOp::NONE; ///< Atomic RMW operation (NONE for regular loads/stores).
  bool lds_dst = false;                ///< Buffer load with LDS bit: write to LDS, not VGPRs.
  uint32_t lds_base = 0;               ///< M0 value for LDS-destination buffer loads.
  uint64_t issue_pc = 0;               ///< PC at which the instruction was issued (debug).
  std::vector<uint8_t> response_data;
  std::vector<uint8_t> store_data;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MEM_STATE_H_
