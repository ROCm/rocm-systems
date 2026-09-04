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

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// gfx1250 cluster async-to-LDS uses the low M0 bits as a destination
/// workgroup-rank mask. Dispatch validation keeps cluster size within this
/// architectural mask width.
constexpr uint32_t kClusterMulticastMaskBits = 16;
constexpr uint32_t kClusterMulticastMask = (1u << kClusterMulticastMaskBits) - 1u;

constexpr uint32_t cluster_multicast_rank_mask(uint32_t cluster_rank) {
  return cluster_rank < kClusterMulticastMaskBits ? (1u << cluster_rank) : 0u;
}

/// @brief Pipeline routing tags for AMDGPU memory instructions.
enum MemPipelineTag : uint8_t {
  SCALAR_MEM = 1,
  GLOBAL_MEM = 2,
  LOCAL_MEM = 3,
};

/// @brief Atomic read-modify-write operation type.
enum class AtomicOp : uint8_t {
  NONE = 0,       ///< Not an atomic operation.
  SWAP,           ///< Exchange.
  CMPSWAP,        ///< Compare-and-swap (data[0] = src, data[1] = cmp).
  MSKOR,          ///< Masked OR (data[0] = mask, data[1] = src).
  ADD,            ///< Atomic add.
  SUB,            ///< Atomic subtract (mem - data).
  RSUB,           ///< Atomic reverse subtract (data - mem).
  SMIN,           ///< Signed minimum.
  UMIN,           ///< Unsigned minimum.
  SMAX,           ///< Signed maximum.
  UMAX,           ///< Unsigned maximum.
  AND,            ///< Bitwise AND.
  OR,             ///< Bitwise OR.
  XOR,            ///< Bitwise XOR.
  INC,            ///< Increment (wrapping).
  DEC,            ///< Decrement (wrapping).
  FADD,           ///< Floating-point add.
  FMIN,           ///< Floating-point minimum.
  FMAX,           ///< Floating-point maximum.
  APPEND,         ///< LDS append counter.
  CONSUME,        ///< LDS consume counter.
  BARRIER_ARRIVE, ///< LDS barrier-arrive state update.
};

/// @brief Dynamic pipeline state for scalar memory instructions (SMEM).
struct ScalarMemState : DynamicInstState {
  ScalarMemState() { tag_ = SCALAR_MEM; }
  uint64_t addr = 0;
  /// Architectural destination resolved and range-checked at issue time.
  ScalarRegisterRange dst_register;
  uint32_t num_dwords = 0;
  uint32_t elem_size = 4;
  bool sign_extend = false;
  bool is_load = true;
  Mtype mtype = Mtype::RW;
  WaitCounterType wait_counter_type = WaitCounterType::LGKMCNT;
  uint32_t response_data[16] = {};
  uint32_t store_data[16] = {};
};

/// @brief Per-element vector-memory lane masks with inline storage for the
/// common one-to-four-element access widths.
class ElementLaneMasks {
public:
  static constexpr size_t kInlineCapacity = 4;

  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] size_t size() const { return size_; }

  void clear() {
    size_ = 0;
    overflow_.clear();
  }

  void assign(size_t count, uint64_t value) {
    if (count <= kInlineCapacity) {
      overflow_.clear();
      for (size_t i = 0; i < count; ++i)
        inline_[i] = value;
      size_ = count;
      return;
    }
    overflow_.assign(count, value);
    size_ = count;
  }

  uint64_t &operator[](size_t index) {
    assert(index < size_);
    return data()[index];
  }

  const uint64_t &operator[](size_t index) const {
    assert(index < size_);
    return data()[index];
  }

  [[nodiscard]] std::span<const uint64_t> view() const { return {data(), size_}; }

private:
  [[nodiscard]] uint64_t *data() {
    return size_ <= kInlineCapacity ? inline_.data() : overflow_.data();
  }

  [[nodiscard]] const uint64_t *data() const {
    return size_ <= kInlineCapacity ? inline_.data() : overflow_.data();
  }

  std::array<uint64_t, kInlineCapacity> inline_{};
  std::vector<uint64_t> overflow_;
  size_t size_ = 0;
};

/// @brief Dynamic pipeline state for vector memory instructions
/// (FLAT, MUBUF, MTBUF, DS).
struct VectorMemState : DynamicInstState {
  VectorMemState(MemPipelineTag pipeline) {
    tag_ = pipeline;
    wait_counter_type = (pipeline == LOCAL_MEM) ? WaitCounterType::LGKMCNT : WaitCounterType::VMCNT;
  }
  std::array<uint64_t, 64> per_lane_addr = {};
  uint64_t lane_mask = 0;
  /// Optional per-element lane validity for untyped DWORD-component bounds.
  /// Empty means every element uses lane_mask; otherwise the container has
  /// exactly num_elems masks and lane_mask is their union.
  ElementLaneMasks element_lane_masks;
  uint64_t exec_mask = 0; ///< Effective issue mask set by address calculation. This normally
                          ///< snapshots EXEC, but ISA exceptions may replace it (for example,
                          ///< CDNA5 DS transpose loads use an all-lanes mask), while
                          ///< architecturally ignored accesses clear it. Writeback zeroes OOB
                          ///< lanes (exec_mask & ~lane_mask).
  uint32_t wf_size = 64;  ///< Wavefront width (set from wavefront's wf_size()).
  uint32_t dst_reg_base = 0;
  uint32_t elem_size = 0;
  uint32_t num_elems = 0;
  bool is_load = true;
  Mtype mtype = Mtype::RW;
  WaitCounterType wait_counter_type = WaitCounterType::VMCNT;
  bool non_temporal = false;
  // Keep this outside Mtype: cluster loads force only the request-side vector
  // L1 lookup to miss, while mtype must still preserve the instruction/PTE
  // cacheability and response policy used by the downstream memory path.
  bool request_force_l1_bypass = false;
  bool sign_extend = false;
  // Scratch (private) accesses store data in the hardware dword-interleaved
  // ("swizzled") layout that rocm-dbgapi reads: consecutive dwords of a lane's
  // private space are lane_count*4 bytes apart, not contiguous. When
  // scratch_swizzle is set, per_lane_addr holds the swizzled address of element
  // 0 and scratch_addr_stride (= lane_count * sizeof(uint32_t)) is the per-element
  // destination-address stride; the register/LDS buffer indexing is unchanged.
  // See rocm-dbgapi memory.cpp private_swizzled conversion.
  // FLAT routing is per lane: one wave can mix private-aperture lanes with
  // global ones. scratch_lane_mask records exactly which lanes were swizzled,
  // so the stride is applied to those and not to their global neighbours.
  // For dedicated SCRATCH ops every active lane is private and this equals
  // lane_mask.
  bool scratch_swizzle = false;
  uint64_t scratch_lane_mask = 0;
  uint32_t scratch_addr_stride = 0;
  // Low bits of the uniform address contribution applied after swizzling. The
  // cache walker subtracts this contribution when locating logical dword
  // boundaries, while per_lane_addr remains the actual first-byte address.
  uint32_t scratch_addr_base_offset = 0;
  bool d16_hi = false; ///< D16_HI load: write upper 16 bits; preserve or zero lower per SRAM ECC.
  bool d16_lo = false; ///< D16 load: write lower 16 bits; preserve or zero upper per SRAM ECC.
  AtomicOp atomic_op = AtomicOp::NONE; ///< Atomic RMW operation (NONE for regular loads/stores).
  bool lds_dst = false;                ///< Buffer load with LDS bit: write to LDS, not VGPRs.
  /// Reference LDS address for LDS-destination loads. For ordinary LDS-dst
  /// paths this may include the lane-0 destination offset. For cluster
  /// multicast this must be exactly Wavefront::lds_base(), the source WG
  /// allocation base; per-lane destination offsets are carried in
  /// per_lane_lds_addr.
  uint32_t lds_base = 0;
  bool lds_per_lane_addr = false; ///< Use per_lane_lds_addr for LDS destination addresses.
  std::array<uint32_t, 64> per_lane_lds_addr = {};
  bool cluster_multicast = false;  ///< Cluster async-to-LDS load: multicast LDS writes by M0 mask.
  uint32_t cluster_mcast_mask = 0; ///< Cluster workgroup destination mask captured at issue time.
  uint64_t issue_pc = 0;           ///< PC at which the instruction was issued (debug).
  // Snapshot dispatch identity at issue time for deferred trace output. The CU
  // is stable for the slot, so its path is materialized only inside log lambdas.
  uint32_t wg_id = 0; ///< Workgroup ID (for trace output).
  uint32_t wf_id = 0; ///< Wavefront ID within WG (for trace output).
  std::vector<uint8_t> response_data;
  std::vector<uint8_t> store_data;
  uint8_t transpose = 0; ///< Transpose-load kind (0=none, see ds_transpose.h).

  /// @brief DS dual-access support.
  ///
  /// When ds2_active is true, per_lane_addr holds the first access addresses
  /// and ds2_per_lane_addr holds the second. For ordinary loads and returning
  /// atomics, ds2_dst_reg_base is the VGPR base for the second result and the
  /// two response vectors preserve each access independently. For stores and
  /// atomics, ds2_store_data contains the second access payload.
  bool ds2_active = false;
  std::array<uint64_t, 64> ds2_per_lane_addr = {};
  uint32_t ds2_dst_reg_base = 0;
  std::vector<uint8_t> ds2_store_data;
  std::vector<uint8_t> ds2_response_data;
};

/// @brief Reject a vector-memory instruction before it reaches a memory pipeline.
/// @details Preserve exec_mask so normal load completion semantics treat every
/// denied lane as inactive/OOB, while clearing lane addresses and lane_mask so
/// no transaction or store reaches memory.
inline void reject_vector_memory_access(VectorMemState &state) {
  state.lane_mask = 0;
  state.per_lane_addr.fill(0);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_MEM_STATE_H_
