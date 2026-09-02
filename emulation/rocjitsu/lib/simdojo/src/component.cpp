// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/sim/component.h"

#include "simdojo/sim/simulation.h"

#include <stdexcept>

namespace simdojo {

void Component::schedule_event(Event *event, Tick timestamp, std::unique_ptr<Message> message) {
  engine_->schedule_event(event, timestamp, std::move(message));
}

bool Component::attached() const {
  return engine_ != nullptr && partition_id_ < engine_->num_contexts();
}

bool Component::schedule_wake(Event *event, Tick timestamp) {
  return attached() && engine_->schedule_wake(event, timestamp);
}

bool Component::wake_pending(const Event &event) const {
  return attached() && engine_->wake_pending(event);
}

Tick Component::current_tick() const {
  return attached() ? engine_->context(partition_id_).current_tick() : 0;
}

Port *Component::add_port(std::unique_ptr<Port> port) {
  Port *raw = port.get();
  ports_.push_back(std::move(port));
  return raw;
}

Port *Component::find_port(PortID port_id) const {
  for (auto &p : ports_) {
    if (p->port_id() == port_id)
      return p.get();
  }
  return nullptr;
}

Component *CompositeComponent::add_child(std::unique_ptr<Component> child) {
  assert(child->parent() == nullptr && "Component already has a parent - remove it first");
  child->set_parent(this);
  child->set_depth(depth() + 1);
  Component *raw = child.get();
  children_.push_back(std::move(child));
  return raw;
}

void CompositeComponent::adopt_children(CompositeComponent &donor) {
  for (auto &child : donor.children_) {
    child->set_parent(this);
    child->set_depth(depth() + 1);
    children_.push_back(std::move(child));
  }
  donor.children_.clear();
}

Component *CompositeComponent::find_child(const std::string &name) const {
  for (auto &c : children_) {
    if (c->name() == name)
      return c.get();
  }
  return nullptr;
}

void CompositeComponent::collect_components(std::vector<Component *> &out) {
  out.push_back(this);
  for (auto &child : children_) {
    auto *composite = dynamic_cast<CompositeComponent *>(child.get());
    if (composite != nullptr) {
      composite->collect_components(out);
    } else {
      out.push_back(child.get());
    }
  }
}

uint32_t CompositeComponent::num_descendants() const {
  uint32_t count = 0;
  for (auto &child : children_) {
    count++;
    auto *composite = dynamic_cast<const CompositeComponent *>(child.get());
    if (composite != nullptr)
      count += composite->num_descendants();
  }
  return count;
}

bool Link::is_cross_partition() const {
  return src_->owner()->partition_id() != dst_->owner()->partition_id();
}

void Link::send_at(std::unique_ptr<Message> msg, Tick ready_tick) {
  if (exec_mode_ == ExecMode::FUNCTIONAL) {
    // FUNCTIONAL mode: call the destination handler synchronously and return.
    // This bypasses LBTS and cross-partition ordering — correct for single-
    // partition functional simulation, but must not be used across partitions
    // in future clocked-mode configurations where LBTS synchronization is needed.
    // There is no simulated time on this path, so ready_tick has nothing to
    // mean; it is deliberately ignored rather than half-honoured. Nor is an
    // engine required: bare links between engineless components are used in
    // functional tests.
    msg->set_latency(latency_);
    Event *port_event = dst_->recv_event();
    if (port_event->has_handler())
      port_event->execute(/*timestamp=*/0, msg.get());
    return;
  }

  const Tick arrival = stamp_for_send(*msg, ready_tick);
  SimulationEngine *engine = src_->owner()->engine();
  Event *port_event = dst_->recv_event();

  if (is_cross_partition()) {
    engine->send_cross_partition(src_->owner()->partition_id(), dst_->owner()->partition_id(),
                                 port_event, arrival, std::move(msg));
  } else {
    engine->schedule_event(port_event, arrival, std::move(msg));
  }
}

Tick Link::depart_now() const {
  SimulationEngine *engine = src_->owner()->engine();
  const PartitionID pid = src_->owner()->partition_id();
  // Whether the partition context exists, not whether create() has finished:
  // create() runs every component's initialize() hook before it marks the
  // engine created, and a component priming a link from that hook has a
  // perfectly good tick to depart at. This still refuses a send against the
  // freed contexts shutdown() leaves, which is the case that matters.
  if (engine == nullptr || pid >= engine->num_contexts())
    throw std::logic_error("a clocked link requires a component attached to a live engine");
  return engine->context(pid).current_tick();
}

Tick Link::depart_now_or_zero() const {
  // Settled here rather than inside depart_now() so that a buffered functional
  // link -- which stamps but never schedules -- needs no engine at all.
  return exec_mode_ == ExecMode::FUNCTIONAL ? Tick{0} : depart_now();
}

Tick Link::stamp_for_send(Message &message, Tick ready_tick) const {
  // Resolved first so that a link whose engine is gone reports that, rather
  // than blaming a tick that was only ever going to be compared against a
  // clock this link no longer has.
  const Tick now = depart_now_or_zero();

  if (ready_tick == TICK_MAX) {
    // The "no such tick" value. A sender that got here from a saturated
    // deadline has computed a completion it cannot meet; delivering the
    // message now, which is what the unstamped sentinel used to mean, would
    // turn "never" into "immediately".
    throw std::invalid_argument("a link cannot carry a message departing at TICK_MAX");
  }
  if (ready_tick < now)
    throw std::invalid_argument("a link cannot carry a message departing before the sender's tick");

  // Overwritten, not consulted: a forwarded message still holds the departure
  // stamp of the hop it arrived on, and that is a fact about the past rather
  // than a request about this send.
  message.set_timestamp(ready_tick);
  message.set_latency(latency_);

  const Tick arrival = message.arrival_tick();
  if (arrival == TICK_MAX) {
    // The departure was representable but the crossing is not. A message
    // landing on TICK_MAX is indistinguishable from no message at all: LBTS
    // advances straight past it, and a buffered one strands itself at the head
    // of a queue its owner has been told is empty.
    throw std::invalid_argument("a link cannot carry a message arriving at TICK_MAX");
  }
  return arrival;
}

} // namespace simdojo
