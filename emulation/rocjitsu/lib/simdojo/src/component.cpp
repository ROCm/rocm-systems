// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/sim/component.h"

#include "simdojo/sim/simulation.h"

#include <cassert>

namespace simdojo {

void Component::schedule_event(Event *event, Tick timestamp, std::unique_ptr<Message> message) {
  engine_->schedule_event(event, timestamp, std::move(message));
}

bool Component::deschedule_event(Event *event, Tick timestamp) {
  return engine_->deschedule(event, timestamp);
}

Tick Component::current_tick() const {
  // shutdown() clears the partition contexts but leaves every component holding
  // its engine, so a non-null engine_ alone does not mean the context exists.
  if (engine_ == nullptr || partition_id_ >= engine_->num_contexts())
    return 0;
  return engine_->context(partition_id_).current_tick();
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

util::Result Link::send_at(std::unique_ptr<Message> msg, Tick ready_tick) {
  // Validated before either delivery path, so a functional link refuses what a
  // clocked one refuses.
  const util::FailureOr<Tick> arrival = stamp_for_send(*msg, ready_tick);
  if (arrival.failed())
    return util::Result::failure();

  if (exec_mode_ == ExecMode::FUNCTIONAL) {
    // Delivered synchronously, so there is no simulated time on this path: the
    // message departs and is handled at tick zero, whatever tick the sender
    // named. The header says so before the handler reads it.
    msg->set_timestamp(0);
  }
  dst_->recv(std::move(msg), arrival.value(), src_->owner()->partition_id());
  return util::Result::success();
}

void Port::recv(std::unique_ptr<Message> msg, Tick arrival, PartitionID from) {
  if (link_->exec_mode() == ExecMode::FUNCTIONAL) {
    // Calls the handler synchronously. This bypasses LBTS and cross-partition
    // ordering -- correct for single-partition functional simulation, but not
    // across partitions in clocked configurations where LBTS synchronization is
    // needed. Nor is an engine required: bare links between engineless
    // components are used in functional tests.
    if (recv_event_.has_handler())
      recv_event_.execute(/*timestamp=*/0, msg.get());
    return;
  }

  SimulationEngine *engine = owner_->engine();
  const PartitionID here = owner_->partition_id();
  if (from != here)
    engine->send_cross_partition(from, here, &recv_event_, arrival, std::move(msg));
  else
    engine->schedule_event(&recv_event_, arrival, std::move(msg));
}

Tick Link::depart_now() const {
  SimulationEngine *engine = src_->owner()->engine();
  const PartitionID pid = src_->owner()->partition_id();
  // Whether the partition context exists, not whether create() has finished:
  // create() runs every component's initialize() hook before it marks the
  // engine created, and a component priming a link from that hook has a
  // perfectly good tick to depart at. This still catches a send against the
  // freed contexts shutdown() leaves, which is the case that matters.
  assert(engine != nullptr && pid < engine->num_contexts() &&
         "a clocked link requires a component attached to a live engine");
  return engine->context(pid).current_tick();
}

Tick Link::depart_now_or_zero() const {
  // Settled here rather than inside depart_now() so that a functional link --
  // which never schedules -- needs no engine at all.
  return exec_mode_ == ExecMode::FUNCTIONAL ? Tick{0} : depart_now();
}

util::FailureOr<Tick> Link::stamp_for_send(Message &message, Tick ready_tick) const {
  // A saturated deadline is the "no such tick" value: the sender has computed
  // a completion it cannot meet, and delivering the message now, which is what
  // the unstamped sentinel used to mean, would turn "never" into "immediately".
  if (ready_tick == TICK_MAX)
    return util::Result::failure();
  assert(ready_tick >= depart_now_or_zero() &&
         "a link cannot carry a message departing before the sender's tick");

  // Overwritten, not consulted: a forwarded message still holds the departure
  // stamp of the hop it arrived on, and that is a fact about the past rather
  // than a request about this send.
  message.set_timestamp(ready_tick);
  message.set_latency(latency_);

  // The departure was representable but the crossing may not be. A message
  // landing on TICK_MAX is indistinguishable from no message at all: LBTS
  // advances straight past it, and a buffered one strands itself at the head
  // of a queue its owner has been told is empty.
  const Tick arrival = message.arrival_tick();
  if (arrival == TICK_MAX)
    return util::Result::failure();
  return arrival;
}

} // namespace simdojo
