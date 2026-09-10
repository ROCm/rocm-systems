// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_access_observation.h
/// @brief What one AMDGPU memory instruction turned out to be, after routing.

#pragma once

#include "rocjitsu/vm/amdgpu/atomic_op.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace rocjitsu::amdgpu {

/// @brief The pipeline a memory instruction was actually issued to.
/// @details The enumerators deliberately share MemPipelineTag's numbering. The
/// pairing is pinned by static_assert where the two headers meet, in
/// compute_unit.cpp, so a new pipeline tag cannot gain a route silently.
enum class MemoryRoute : uint8_t {
  UNKNOWN, ///< Not issued to any pipeline; reported so it is not silently lost.
  SCALAR,  ///< Scalar (constant) memory.
  GLOBAL,  ///< Global, buffer, or scratch.
  LOCAL,   ///< Local data share.
};

/// @brief Report name for a route. Stable; consumers key output on it.
constexpr const char *memory_route_name(MemoryRoute route) {
  switch (route) {
  case MemoryRoute::SCALAR:
    return "scalar";
  case MemoryRoute::GLOBAL:
    return "global";
  case MemoryRoute::LOCAL:
    return "local";
  case MemoryRoute::UNKNOWN:
    break;
  }
  return "unknown";
}

/// @brief One memory instruction as the memory system will see it.
///
/// @details Reported after routing rather than before, which is the whole
/// point. A FLAT access to the shared aperture is decoded as a global
/// instruction and issued to the local pipeline with its addresses rewritten
/// into the workgroup's LDS allocation; an observer that looked before routing
/// would see a global access to an address the memory system never uses, and
/// would charge it against the wrong cache with the wrong hit rate.
///
/// Everything here is a fact, not an estimate. Where a fact is missing --
/// a lane whose address could not be resolved, a route with no pipeline --
/// it is reported as such rather than filled in.
///
/// The spans point into the instruction's own state and are valid only for
/// the duration of the callback. A consumer that keeps an observation must
/// copy them.
struct MemoryAccessObservation {
  /// @brief Instruction mnemonic; points at static storage.
  std::string_view mnemonic;
  uint64_t pc = 0;              ///< PC of the issuing wavefront.
  uint32_t compute_unit_id = 0; ///< Component id of the compute unit that issued it.
  uint32_t dispatch_id = 0;
  uint32_t queue_id = 0; ///< Queue the dispatch was submitted to.
  uint32_t workgroup_id = 0;
  uint32_t wavefront_id = 0; ///< Wavefront slot within the compute unit.
  uint32_t process_id = 0;   ///< VMID of the address space the addresses live in.

  /// @brief The pipeline that took the access.
  ///
  /// @details UNKNOWN means no pipeline did. Every field below is then left at
  /// its default and means nothing; only the identity above and the mnemonic
  /// describe the instruction. Check this before reading anything else.
  MemoryRoute route = MemoryRoute::UNKNOWN;
  /// @brief Whether a FLAT access was rewritten from global into LDS.
  ///
  /// @details True only for the aperture case: the instruction decoded as
  /// global, and both its route and its addresses were changed. A consumer
  /// counting "how much of this kernel is really LDS traffic" needs to
  /// separate these from instructions that were LDS to begin with.
  bool normalized_to_local = false;

  /// @brief Whether the access returns a value to registers.
  ///
  /// @details Not a statement about direction when @ref atomic_op is set: a
  /// returning atomic has this true and a non-returning one false, and both
  /// read and write the line. Split read from write traffic on the pair, not
  /// on this flag alone.
  bool is_load = true;
  /// @brief Atomic read-modify-write operation, NONE for a plain access.
  ///
  /// @details APPEND and CONSUME are wave-collapsing: the whole wavefront
  /// performs one 4-byte LDS read-modify-write at the first valid lane's
  /// address, and the per-lane addresses are all that address. Costing them
  /// per requesting lane over-reports by the lane count.
  AtomicOp atomic_op = AtomicOp::NONE;
  Mtype mtype = Mtype::RW;
  /// @brief The counter this access will post to, which is what an s_waitcnt
  ///        naming that counter will wait on. Set by routing, not by decode:
  ///        an aperture-rewritten FLAT posts to LGKMCNT, not VMCNT.
  WaitCounterType wait_counter = WaitCounterType::VMCNT;

  /// @brief Lanes in the issuing wavefront; 1 for scalar, 0 when the route is
  ///        UNKNOWN and there is no wavefront width to report.
  uint32_t wavefront_size = 0;
  uint32_t element_size_bytes = 0; ///< Bytes per element.
  /// @brief Elements each participating lane moves, through one address set.
  ///
  /// @details A DS dual access has two address sets and moves this many
  /// through each; see @ref secondary_addresses.
  uint32_t elements_per_lane = 0;

  /// @brief Lanes the instruction issued with.
  ///
  /// @details Normally a snapshot of EXEC, but an ISA exception may replace it:
  /// a CDNA5 DS transpose load issues with every lane regardless of EXEC.
  uint64_t active_lane_mask = 0;
  /// @brief Lanes whose address resolved to something the memory system can
  ///        use. A lane that is active but not valid went out of bounds or
  ///        through an unmapped aperture.
  uint64_t valid_lane_mask = 0;
  /// @brief Lanes that actually produce a memory request.
  ///
  /// @details Differs from @ref valid_lane_mask only for the wave64 transpose
  /// loads that issue through the low half -- a `WMMA_TR_B8`, or a global
  /// `TR16_B128` -- where one lane fetches on behalf of several. Every other
  /// access, transpose or not, requests from every valid lane.
  uint64_t request_lane_mask = 0;
  /// @brief Lanes whose addresses are swizzled scratch; see
  ///        @ref scratch_element_stride_bytes. A FLAT wave can mix these with
  ///        global lanes.
  uint64_t scratch_lane_mask = 0;
  /// @brief Byte stride between consecutive elements of a scratch lane. Zero
  ///        when the access is contiguous, which every non-scratch access is.
  ///
  /// @details Reported from the instruction's own state, so it is set on an
  /// atomic whose lanes are swizzled scratch even though the atomic path
  /// applies no stride.
  uint32_t scratch_element_stride_bytes = 0;

  bool non_temporal = false;    ///< Documented not to allocate in a cache honouring it.
  bool force_l1_bypass = false; ///< Request-side first-level lookup forced to miss.
  /// @brief Global load whose result is written to LDS rather than to VGPRs.
  ///
  /// @details The addresses reported here are the global side only. The LDS
  /// side -- the destination addresses, and the cluster multicast fan-out that
  /// can replicate the write to several workgroups' allocations -- is not part
  /// of this observation, so an LDS bandwidth or bank-conflict model cannot be
  /// built from these accesses.
  bool lds_destination = false;

  /// @brief Address each lane accesses, @ref wavefront_size entries.
  ///
  /// @details Only entries selected by @ref valid_lane_mask are meaningful.
  std::span<const uint64_t> addresses;
  /// @brief Per-element lane validity, when an access has narrower bounds for
  ///        later elements than for earlier ones. Empty means every element
  ///        uses @ref valid_lane_mask.
  std::span<const uint64_t> element_lane_masks;
  /// @brief Addresses of the second access of a DS dual-access instruction.
  ///        Empty unless the instruction has one.
  ///
  /// @details The second access moves the same bytes per lane through the same
  /// lanes as the first, so an instruction with a non-empty set here moves
  /// twice @ref bytes_per_lane().
  std::span<const uint64_t> secondary_addresses;

  /// @brief Bytes each participating lane moves through one address set,
  ///        ignoring scratch swizzling. Double it when @ref
  ///        secondary_addresses is non-empty.
  uint64_t bytes_per_lane() const {
    return static_cast<uint64_t>(element_size_bytes) * elements_per_lane;
  }

  /// @brief Every lane the wavefront has, whether it participated or not.
  uint64_t wavefront_lane_mask() const {
    return wavefront_size >= 64 ? ~uint64_t{0} : (uint64_t{1} << wavefront_size) - 1;
  }

  /// @brief Lanes EXEC masked off. They cost no traffic, but a model that
  ///        counted them would report a wave64 access for a wave that had
  ///        two lanes on.
  uint64_t inactive_lane_mask() const { return wavefront_lane_mask() & ~active_lane_mask; }

  /// @brief Lanes that issued but whose address the memory system will not use.
  ///
  /// @details Out of bounds, through an unmapped aperture, or belonging to an
  /// access denied wholesale -- a rejected access clears every valid lane
  /// while leaving the active ones, so all of them land here. Enumerated
  /// rather than approximated: a model has no basis for costing these, and the
  /// honest thing is to say how many there were.
  uint64_t unknown_lane_mask() const { return active_lane_mask & ~valid_lane_mask; }
};

} // namespace rocjitsu::amdgpu
