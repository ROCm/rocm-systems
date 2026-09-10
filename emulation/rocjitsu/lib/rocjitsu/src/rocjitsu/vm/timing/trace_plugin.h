// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file trace_plugin.h
/// @brief The execution plugin that feeds the timing model, and nothing else.

#ifndef ROCJITSU_VM_TIMING_TRACE_PLUGIN_H_
#define ROCJITSU_VM_TIMING_TRACE_PLUGIN_H_

#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/timing/trace_source.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <utility>

namespace rocjitsu::timing {

/// @brief Translates execution hooks into trace events. It does not simulate.
///
/// @details A collector, deliberately: it reads facts off the functional side
/// and hands them to the trace source, and it owns no clock, no queue, and no
/// notion of cost. Everything that decides how long something takes lives
/// downstream of the source, in the modelled topology.
///
/// It declares serial hot hooks. Its state -- the per-wavefront instruction
/// ordinals, and the source's stream position -- is shared across every
/// wavefront the group observes, and a stream assembled from unsynchronised
/// concurrent callbacks would not be a stream.
class TimingTracePlugin final : public ExecutionPlugin {
public:
  explicit TimingTracePlugin(TimingTraceSource &source);

  bool requires_serial_hot_hooks() const override { return true; }
  bool observes_memory_routing() const override { return true; }
  /// @brief Register reads say nothing about cost that the instruction itself
  ///        does not, and they are the highest-frequency callback there is.
  bool observes_sgpr_reads() const override { return false; }

  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;
  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;
  /// @details Before rather than after, for two reasons. A terminator retires
  /// without an after-execute callback -- the wavefront has halted by then --
  /// so a stream built on the later hook is missing exactly one instruction per
  /// wavefront, silently. And the lanes that *executed* an instruction are the
  /// ones EXEC held before it ran, which is what an instruction that writes
  /// EXEC makes visible.
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;
  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access) override;
  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) override;

  /// @brief Facts the model has no term for, counted rather than hidden.
  uint64_t unsupported() const { return unsupported_; }

  /// @brief Facts the source would not take, counted rather than hidden.
  ///
  /// @details A refusal is a fact that did not reach the model. Nothing else
  /// on this side of the boundary would ever mention it.
  uint64_t refused() const { return refused_; }

private:
  /// @brief Hand @p event to the source, counting a refusal.
  void submit(TraceEvent event);
  /// @brief Identity of @p wf, as functional execution has it.
  static TraceIdentity identity_of(amdgpu::Wavefront &wf);
  /// @brief Key into @ref ordinals_ for a compute unit and wavefront slot.
  static uint64_t ordinal_key(uint32_t compute_unit_id, uint32_t wavefront_id) {
    return (static_cast<uint64_t>(compute_unit_id) << 32) | wavefront_id;
  }

  /// @brief The next instruction ordinal for @p identity's slot.
  uint64_t next_ordinal(const TraceIdentity &identity);

  /// @brief The ordinal most recently handed out for @p identity's slot, which
  ///        is the instruction currently executing.
  uint64_t current_ordinal(const TraceIdentity &identity) const;

  TimingTraceSource &source_;
  /// @brief Per-wavefront instruction counts, keyed by compute unit and slot.
  ///        Erased when the wavefront halts, so this follows occupancy rather
  ///        than the length of the run.
  std::unordered_map<uint64_t, uint64_t> ordinals_;
  /// @brief A dispatch shape and when it arrived.
  struct AnnouncedShape {
    uint64_t arrival = 0;
    KernelDispatchInfo info;
  };

  /// @brief Dispatch shapes seen at packet-processed time, replayed into the
  ///        stream when the dispatch actually begins. The packet is parsed
  ///        ahead of execution, so the shape arrives before it is wanted.
  ///
  /// @details Bounded: a packet whose dispatch never begins -- the run ended,
  /// or the packet was cancelled -- would otherwise leave its shape, and the
  /// two strings in it, here for good.
  std::unordered_map<uint32_t, AnnouncedShape> announced_;
  /// @brief The order shapes arrived, so the oldest can be dropped.
  ///
  /// @details By arrival rather than by id. Dispatch ids do not simply climb:
  /// CommandProcessor::allocate_dispatch_id restarts at its XCD's base when the
  /// counter wraps, so the smallest id in the map can be the newest shape in
  /// it -- and evicting by id would then drop the shape of the dispatch about
  /// to begin, reporting a kernel whose grid was in fact observed as one whose
  /// packet was never seen. The arrival stamp also tells a spent entry, whose
  /// shape a dispatch-begin already collected, from a live one.
  std::deque<std::pair<uint32_t, uint64_t>> announced_order_;
  /// @brief Shapes seen, ever. Names which arrival a window entry stands for.
  uint64_t announced_arrival_ = 0;
  /// @brief Most announced-but-unstarted dispatches kept. Deep enough to cover
  /// however far ahead the command processor parses, which is a queue's worth,
  /// not a run's.
  static constexpr std::size_t kMaxAnnounced = 256;
  uint64_t unsupported_ = 0;
  uint64_t refused_ = 0;
};

} // namespace rocjitsu::timing

#endif // ROCJITSU_VM_TIMING_TRACE_PLUGIN_H_
