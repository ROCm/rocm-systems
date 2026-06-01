// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/sim/component.h"

#include "simdojo/sim/simulation.h"

namespace simdojo {

void Component::run() {
  while (step()) {
  }
}

void Component::schedule_event(Event *event, Tick timestamp, std::unique_ptr<Message> message) {
  engine_->schedule_event(event, timestamp, std::move(message));
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

void CompositeComponent::run() {
  for (auto &child : children_)
    child->run();
}

bool CompositeComponent::step() {
  bool any_active = false;
  for (auto &child : children_) {
    if (child->step())
      any_active = true;
  }
  return any_active;
}

Component *CompositeComponent::add_child(std::unique_ptr<Component> child) {
  assert(child->parent() == nullptr && "Component already has a parent - remove it first");
  child->set_parent(this);
  child->set_depth(depth() + 1);
  Component *raw = child.get();
  children_.push_back(std::move(child));
  return raw;
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

void Link::send(std::unique_ptr<Message> msg) {
  if (exec_mode_ == ExecMode::FUNCTIONAL) {
    // FUNCTIONAL mode: call the destination handler synchronously and return.
    // This bypasses LBTS and cross-partition ordering — correct for single-
    // partition functional simulation, but must not be used across partitions
    // in future clocked-mode configurations where LBTS synchronization is needed.
    msg->set_latency(latency_);
    Event *port_event = dst_->recv_event();
    if (port_event->has_handler())
      port_event->execute(/*timestamp=*/0, msg.get());
    return;
  }

  SimulationEngine *engine = src_->owner()->engine();
  assert(engine != nullptr);

  // Stamp the message with the partition's local processing tick (not GVT).
  // Using GVT would be stale in multi-threaded mode, potentially causing
  // cross-partition messages to arrive before what neighbors have processed.
  // TICK_MAX is the "not yet stamped" sentinel (default in MessageHeader).
  if (msg->header().timestamp == TICK_MAX) {
    PartitionID pid = src_->owner()->partition_id();
    msg->set_timestamp(engine->context(pid).current_tick());
  }
  msg->set_latency(latency_);
  Tick arrival = msg->arrival_tick();

  Event *port_event = dst_->recv_event();

  if (is_cross_partition()) {
    engine->send_cross_partition(src_->owner()->partition_id(), dst_->owner()->partition_id(),
                                 port_event, arrival, std::move(msg));
  } else {
    engine->schedule_event(port_event, arrival, std::move(msg));
  }
}

} // namespace simdojo
