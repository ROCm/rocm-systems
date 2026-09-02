// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/trace_plugin.h"

#include "rocjitsu/isa/instruction.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include <memory>
#include <utility>

namespace rocjitsu::timing {

TimingTracePlugin::TimingTracePlugin(TimingTraceSource &source)
    : ExecutionPlugin("timing_trace"), source_(source) {}

TraceIdentity TimingTracePlugin::identity_of(amdgpu::Wavefront &wf) {
  TraceIdentity identity;
  identity.compute_unit_id = wf.cu().id();
  identity.dispatch_id = wf.dispatch_id();
  identity.queue_id = wf.queue_id();
  identity.process_id = wf.process_id();
  identity.workgroup_id = wf.wg_id();
  identity.wavefront_id = wf.wf_id();
  return identity;
}

void TimingTracePlugin::submit(TraceEvent event) {
  if (!source_.submit(std::move(event)))
    ++refused_;
}

uint64_t TimingTracePlugin::next_ordinal(const TraceIdentity &identity) {
  return ++ordinals_[ordinal_key(identity.compute_unit_id, identity.wavefront_id)];
}

uint64_t TimingTracePlugin::current_ordinal(const TraceIdentity &identity) const {
  const auto it = ordinals_.find(ordinal_key(identity.compute_unit_id, identity.wavefront_id));
  return it == ordinals_.end() ? 0 : it->second;
}

void TimingTracePlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  // Recorded, not emitted. The command processor parses ahead of execution, so
  // a dispatch's shape arrives before the dispatch is anywhere near starting;
  // emitting it here would put it in the stream ahead of the previous
  // dispatch's own end.
  const uint64_t arrival = ++announced_arrival_;
  announced_[info.dispatch_id] = AnnouncedShape{arrival, info};
  announced_order_.emplace_back(info.dispatch_id, arrival);
  if (announced_order_.size() <= kMaxAnnounced)
    return;

  // A shape nothing ever came to collect, dropped oldest-arrival first. Only if
  // this entry is still the one the map holds: a dispatch that began collected
  // its shape already, and an id announced twice has a newer one.
  const auto [oldest_id, oldest_arrival] = announced_order_.front();
  announced_order_.pop_front();
  const auto it = announced_.find(oldest_id);
  if (it != announced_.end() && it->second.arrival == oldest_arrival)
    announced_.erase(it);
}

void TimingTracePlugin::onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
  TraceEvent event;
  event.kind = TraceEventKind::DISPATCH_BEGIN;
  event.identity.dispatch_id = dispatch_id;
  if (auto it = announced_.find(dispatch_id); it != announced_.end()) {
    event.dispatch = std::make_shared<const KernelDispatchInfo>(std::move(it->second.info));
    announced_.erase(it);
  } else {
    // A dispatch that started without its packet being observed. The model can
    // still time its instructions, but it cannot know the grid it was supposed
    // to cover, so it must not report coverage for it.
    event.supported = false;
    event.unsupported_reason = "dispatch began without an observed packet";
    ++unsupported_;
  }
  submit(std::move(event));
}

void TimingTracePlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  TraceEvent event;
  event.kind = TraceEventKind::DISPATCH_END;
  event.identity.dispatch_id = dispatch_id;
  submit(std::move(event));
}

void TimingTracePlugin::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
  TraceEvent event;
  event.kind = TraceEventKind::WAVE_BEGIN;
  event.identity = identity_of(wf);
  event.pc = wf.pc;
  event.active_lane_mask = wf.exec();
  // The slot starts this wavefront at one. Cleared on placement rather than
  // only on halt because a wavefront can leave a slot without halting: the
  // clustered-dispatch rollback frees its committed peers through
  // ComputeUnitCore::abort_workgroup(), which fires no completion hook. The
  // erase on halt is then a bound on how much this map holds, not the thing
  // that makes an ordinal mean what it says.
  ordinals_.erase(ordinal_key(event.identity.compute_unit_id, event.identity.wavefront_id));
  submit(std::move(event));
}

void TimingTracePlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  TraceEvent event;
  event.kind = TraceEventKind::WAVE_END;
  event.identity = identity_of(wf);
  event.pc = wf.pc;
  const uint64_t slot = ordinal_key(event.identity.compute_unit_id, event.identity.wavefront_id);
  if (auto it = ordinals_.find(slot); it != ordinals_.end()) {
    event.instruction_ordinal = it->second;
    // The slot is about to be handed to another wavefront, whose ordinals
    // start again from one.
    ordinals_.erase(it);
  }
  submit(std::move(event));
}

void TimingTracePlugin::onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                         amdgpu::Wavefront &wf) {
  TraceEvent event;
  event.kind = TraceEventKind::INSTRUCTION;
  event.identity = identity_of(wf);
  event.pc = pc;
  event.mnemonic = inst.mnemonic();
  event.active_lane_mask = wf.exec();
  event.instruction_ordinal = next_ordinal(event.identity);
  submit(std::move(event));
}

void TimingTracePlugin::onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access) {
  TraceEvent event;
  event.kind = TraceEventKind::MEMORY;
  event.identity.compute_unit_id = access.compute_unit_id;
  event.identity.dispatch_id = access.dispatch_id;
  event.identity.process_id = access.process_id;
  event.identity.workgroup_id = access.workgroup_id;
  event.identity.queue_id = access.queue_id;
  event.identity.wavefront_id = access.wavefront_id;
  // The instruction currently issuing on that slot: inside a loop the PC
  // repeats, so it cannot say which of a wavefront's accesses this is.
  event.instruction_ordinal = current_ordinal(event.identity);
  event.pc = access.pc;
  event.mnemonic = access.mnemonic;
  event.active_lane_mask = access.active_lane_mask;
  event.memory = std::make_shared<const MemoryFacts>(MemoryFacts::from(access));
  if (access.route == amdgpu::MemoryRoute::UNKNOWN) {
    // Routing found no pipeline for it, so there is no level of the hierarchy
    // to charge it against. Emitted rather than dropped: a kernel with traffic
    // nobody can account for should say so.
    event.supported = false;
    event.unsupported_reason = "memory access was routed to no pipeline";
    ++unsupported_;
  }
  submit(std::move(event));
}

void TimingTracePlugin::onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
  if (wavefronts.empty())
    return;
  TraceEvent event;
  event.kind = TraceEventKind::BARRIER;
  event.identity = identity_of(*wavefronts.front());
  event.participants.reserve(wavefronts.size());
  for (amdgpu::Wavefront *wf : wavefronts)
    event.participants.push_back(identity_of(*wf));
  submit(std::move(event));
}

} // namespace rocjitsu::timing
