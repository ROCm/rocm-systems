// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file trace_event.h
/// @brief The facts functional execution reports to a timing model.

#ifndef ROCJITSU_VM_TIMING_TRACE_EVENT_H_
#define ROCJITSU_VM_TIMING_TRACE_EVENT_H_

#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/plugins/memory_access_observation.h"

#include "simdojo/sim/message.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rocjitsu::timing {

/// @brief What kind of fact an event reports.
enum class TraceEventKind : uint8_t {
  DISPATCH_BEGIN, ///< A dispatch's workgroups are about to be placed.
  DISPATCH_END,   ///< Every workgroup of a dispatch has completed.
  WAVE_BEGIN,     ///< A wavefront has been placed and is about to issue.
  WAVE_END,       ///< A wavefront has halted.
  INSTRUCTION,    ///< One instruction issued.
  MEMORY,         ///< One memory instruction, as routing left it.
  BARRIER,        ///< A workgroup's barrier released.
};

/// @brief Report name for a kind. Stable; consumers key output on it.
constexpr const char *trace_event_kind_name(TraceEventKind kind) {
  switch (kind) {
  case TraceEventKind::DISPATCH_BEGIN:
    return "dispatch_begin";
  case TraceEventKind::DISPATCH_END:
    return "dispatch_end";
  case TraceEventKind::WAVE_BEGIN:
    return "wave_begin";
  case TraceEventKind::WAVE_END:
    return "wave_end";
  case TraceEventKind::INSTRUCTION:
    return "instruction";
  case TraceEventKind::MEMORY:
    return "memory";
  case TraceEventKind::BARRIER:
    return "barrier";
  }
  return "unknown";
}

/// @brief Where on the machine a fact happened.
///
/// @details The identity functional execution actually used, not a modelled
/// one. A timing model that reassigned work to its own idea of a compute unit
/// would model a different machine from the one that produced the numbers it
/// is being compared against.
struct TraceIdentity {
  /// @brief Component id of the compute unit the work actually ran on.
  ///
  /// @details Unique across the whole topology, so it names the device, the
  /// die, and the shader engine as well: a model that partitions by die
  /// resolves the mapping once from the topology rather than having every
  /// event carry a redundant copy of it. The identity is functional
  /// execution's own -- a model that reassigned work to its own idea of a
  /// compute unit would be modelling a different machine from the one the
  /// numbers it is compared against came from.
  uint32_t compute_unit_id = 0;
  uint32_t dispatch_id = 0;
  uint32_t queue_id = 0;   ///< Queue the dispatch was submitted to.
  uint32_t process_id = 0; ///< VMID of the issuing process.
  uint32_t workgroup_id = 0;
  uint32_t wavefront_id = 0; ///< Wavefront index within its workgroup.

  bool operator==(const TraceIdentity &) const = default;
};

/// @brief One memory access, copied out of the borrowed observation.
///
/// @details A copy rather than a reference: the observation's spans point into
/// the instruction's own state, which the pipeline reuses as soon as the
/// callback returns. Only the lanes the wavefront actually has are copied, so
/// a wave32 access carries thirty-two addresses rather than sixty-four.
struct MemoryFacts {
  amdgpu::MemoryRoute route = amdgpu::MemoryRoute::UNKNOWN;
  bool normalized_to_local = false;
  bool is_load = true;
  amdgpu::AtomicOp atomic_op = amdgpu::AtomicOp::NONE;
  amdgpu::Mtype mtype = amdgpu::Mtype::RW;
  amdgpu::WaitCounterType wait_counter = amdgpu::WaitCounterType::VMCNT;
  uint32_t wavefront_size = 0;
  uint32_t element_size_bytes = 0;
  uint32_t elements_per_lane = 0;
  uint64_t active_lane_mask = 0;
  uint64_t valid_lane_mask = 0;
  uint64_t request_lane_mask = 0;
  uint64_t scratch_lane_mask = 0;
  uint32_t scratch_element_stride_bytes = 0;
  bool non_temporal = false;
  bool force_l1_bypass = false;
  bool lds_destination = false;
  std::vector<uint64_t> addresses;
  std::vector<uint64_t> element_lane_masks;
  std::vector<uint64_t> secondary_addresses;

  /// @brief Every lane the wavefront has, whether it participated or not.
  uint64_t wavefront_lane_mask() const {
    return wavefront_size >= 64 ? ~uint64_t{0} : (uint64_t{1} << wavefront_size) - 1;
  }
  /// @brief Lanes EXEC masked off. Derived rather than stored, so it cannot
  ///        contradict the masks it is derived from -- including in a
  ///        recording that was edited or partially deserialised.
  uint64_t inactive_lane_mask() const { return wavefront_lane_mask() & ~active_lane_mask; }
  /// @brief Lanes that issued but whose address the memory system will not use.
  uint64_t unknown_lane_mask() const { return active_lane_mask & ~valid_lane_mask; }

  /// @brief Copy the borrowed facts out of @p access.
  static MemoryFacts from(const amdgpu::MemoryAccessObservation &access);
};

/// @brief One fact, with everything a model needs to place it.
struct TraceEvent {
  /// @brief Position in the stream, assigned by the trace source.
  ///
  /// @details Starts at one; zero means "not yet sequenced". Consecutive and
  /// gapless, so a consumer can tell a dropped event from a reordered one.
  uint64_t sequence = 0;
  TraceEventKind kind = TraceEventKind::INSTRUCTION;
  TraceIdentity identity;

  uint64_t pc = 0;
  /// @brief Which instruction of this wavefront this is, from one.
  ///
  /// @details Carried by MEMORY events too, naming the instruction that made
  /// the access -- inside a loop the PC alone repeats, so it cannot identify
  /// one dynamic access among many.
  uint64_t instruction_ordinal = 0;
  /// @brief Lanes that executed the instruction.
  uint64_t active_lane_mask = 0;
  /// @brief Instruction mnemonic. Copied: the observation's is static storage,
  ///        but a recorded stream outlives the process that produced it.
  std::string mnemonic;

  /// @brief Dispatch shape, on DISPATCH_BEGIN only.
  std::shared_ptr<const KernelDispatchInfo> dispatch;
  /// @brief Memory facts, on MEMORY only.
  std::shared_ptr<const MemoryFacts> memory;
  /// @brief Wavefronts released together, on BARRIER only.
  std::vector<TraceIdentity> participants;

  /// @brief Whether a model can represent this fact at all.
  ///
  /// @details False for something functional execution did that the model has
  /// no term for. The event is still emitted: a model that dropped it would
  /// report a kernel it had modelled completely, and the honest answer is that
  /// it modelled everything except this. See @ref unsupported_reason.
  bool supported = true;
  /// @brief Why @ref supported is false. Empty when it is true.
  std::string unsupported_reason;
};

/// @brief A trace event on its way to a modelled component.
///
/// @details Derives from simdojo::Message rather than introducing a second
/// message base: it travels over ordinary SimDojo ports and links, and the
/// header carries the sequence number so a consumer can correlate without
/// looking inside.
class TraceMessage final : public simdojo::Message {
public:
  explicit TraceMessage(TraceEvent event) : event_(std::move(event)) {
    header_.sequence_num = event_.sequence;
  }

  const TraceEvent &event() const { return event_; }

private:
  TraceEvent event_;
};

} // namespace rocjitsu::timing

#endif // ROCJITSU_VM_TIMING_TRACE_EVENT_H_
