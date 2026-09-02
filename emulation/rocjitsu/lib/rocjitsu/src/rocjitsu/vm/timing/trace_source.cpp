// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/trace_source.h"

#include "simdojo/sim/simulation.h"

#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace rocjitsu::timing {

MemoryFacts MemoryFacts::from(const amdgpu::MemoryAccessObservation &access) {
  MemoryFacts facts;
  facts.route = access.route;
  facts.normalized_to_local = access.normalized_to_local;
  facts.is_load = access.is_load;
  facts.atomic_op = access.atomic_op;
  facts.mtype = access.mtype;
  facts.wait_counter = access.wait_counter;
  facts.wavefront_size = access.wavefront_size;
  facts.element_size_bytes = access.element_size_bytes;
  facts.elements_per_lane = access.elements_per_lane;
  facts.active_lane_mask = access.active_lane_mask;
  facts.valid_lane_mask = access.valid_lane_mask;
  facts.request_lane_mask = access.request_lane_mask;
  facts.scratch_lane_mask = access.scratch_lane_mask;
  facts.scratch_element_stride_bytes = access.scratch_element_stride_bytes;
  facts.non_temporal = access.non_temporal;
  facts.force_l1_bypass = access.force_l1_bypass;
  facts.lds_destination = access.lds_destination;
  facts.addresses.assign(access.addresses.begin(), access.addresses.end());
  facts.element_lane_masks.assign(access.element_lane_masks.begin(),
                                  access.element_lane_masks.end());
  facts.secondary_addresses.assign(access.secondary_addresses.begin(),
                                   access.secondary_addresses.end());
  return facts;
}

TimingTraceSource::TimingTraceSource(std::string name) : simdojo::Component(std::move(name)) {}

void TimingTraceSource::initialize() {
  if (output_ == nullptr) {
    output_ = add_port(std::make_unique<simdojo::Port>("trace_out", /*port_id=*/0, this,
                                                       simdojo::PortDirection::OUT));
  }
}

bool TimingTraceSource::can_send() const {
  if (output_ == nullptr || output_->link() == nullptr)
    return false;
  const simdojo::SimulationEngine *engine = this->engine();
  return engine != nullptr && engine->is_created();
}

bool TimingTraceSource::submit(TraceEvent event) {
  std::lock_guard<std::mutex> lock(mutex_);

  // A dispatch announcing itself again is a new dispatch reusing an id, not a
  // continuation of the one that ended, so a BEGIN is never refused for it.
  const TraceEventKind kind = event.kind;
  const uint32_t dispatch_id = event.identity.dispatch_id;
  if (kind != TraceEventKind::DISPATCH_BEGIN && ended_.contains(dispatch_id)) {
    ++rejected_;
    return false;
  }

  if (!can_send()) {
    ++rejected_;
    return false;
  }

  event.sequence = next_sequence_;
  send_locked(std::move(event));
  // Committed only once the message is actually on the wire. Everything above
  // is a decision; nothing above writes. A send that threw therefore leaves no
  // gap in the stream and no dispatch opened or closed by an event that never
  // went out -- which is what lets submit() be called from a path with nothing
  // to roll back with.
  ++next_sequence_;
  apply_ended_locked(kind, dispatch_id);
  return true;
}

void TimingTraceSource::apply_ended_locked(TraceEventKind kind, uint32_t dispatch_id) {
  if (kind == TraceEventKind::DISPATCH_BEGIN)
    ended_.erase(dispatch_id);
  else if (kind == TraceEventKind::DISPATCH_END)
    mark_ended_locked(dispatch_id);
}

void TimingTraceSource::mark_ended_locked(uint32_t dispatch_id) {
  const uint64_t epoch = ++ended_epoch_;
  // An id already in the window keeps the entry it has: a second END without an
  // intervening BEGIN says nothing new.
  if (!ended_.emplace(dispatch_id, epoch).second)
    return;
  ended_order_.emplace_back(dispatch_id, epoch);
  if (ended_order_.size() <= kEndedWindow)
    return;

  const auto [oldest_id, oldest_epoch] = ended_order_.front();
  ended_order_.pop_front();
  // Only if this entry is the finalisation still in force. A re-announcement
  // drops an id from ended_ without reaching the deque, so the deque can hold a
  // spent entry for an id that has since ended again; evicting the spent one
  // must not take the live guard with it and reopen a dispatch that has ended.
  const auto it = ended_.find(oldest_id);
  if (it != ended_.end() && it->second == oldest_epoch)
    ended_.erase(it);
}

void TimingTraceSource::replay(std::span<const TraceEvent> events) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Onto a fresh source only. Replaying onto one that has already sent
  // something would hand out sequence numbers it had used and wind the counter
  // backwards, and gaplessness is the only thing telling a consumer a dropped
  // event from a reordered one.
  // An empty recording asks for nothing, so it changes nothing -- including
  // the stream position and the dispatches already known to have ended.
  if (events.empty())
    return;
  if (accepted_ != 0 || next_sequence_ != 1)
    throw std::logic_error("a recorded trace can only be replayed into a source that is at rest");
  if (!can_send())
    throw std::logic_error("replay requires a connected output and a live engine");

  uint64_t expected = 1;
  std::unordered_set<uint32_t> ended;
  for (const TraceEvent &event : events) {
    // Checked before anything is sent, so a malformed recording is refused
    // whole rather than halfway through.
    if (event.sequence != expected)
      throw std::invalid_argument("recorded trace is not consecutive from one");
    ++expected;
    if (event.kind == TraceEventKind::DISPATCH_BEGIN)
      ended.erase(event.identity.dispatch_id);
    else if (ended.contains(event.identity.dispatch_id))
      throw std::invalid_argument("recorded trace continues a dispatch after its end");
    if (event.kind == TraceEventKind::DISPATCH_END)
      ended.insert(event.identity.dispatch_id);
  }

  ended_.clear();
  ended_order_.clear();
  for (const TraceEvent &event : events) {
    send_locked(event);
    // Advanced per event rather than once at the end, and through the same
    // helper the live path uses. A send that threw part way then leaves the
    // source describing exactly the prefix that reached the consumer, instead
    // of a stream it claims never started; and the rule about what a BEGIN and
    // an END mean is stated once, so a recording and a live run cannot come to
    // disagree about it.
    next_sequence_ = event.sequence + 1;
    apply_ended_locked(event.kind, event.identity.dispatch_id);
  }
}

void TimingTraceSource::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  ended_.clear();
  ended_order_.clear();
  next_sequence_ = 1;
  accepted_ = 0;
  rejected_ = 0;
}

void TimingTraceSource::send_locked(TraceEvent event) {
  // No departure tick of its own: the link stamps the timing engine's current
  // tick, which is the only clock this plane has.
  output_->send(std::make_unique<TraceMessage>(std::move(event)));
  // Counted after, so a send that threw is not reported as accepted.
  ++accepted_;
}

uint64_t TimingTraceSource::accepted() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return accepted_;
}

uint64_t TimingTraceSource::rejected() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return rejected_;
}

uint64_t TimingTraceSource::next_sequence() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return next_sequence_;
}

} // namespace rocjitsu::timing
