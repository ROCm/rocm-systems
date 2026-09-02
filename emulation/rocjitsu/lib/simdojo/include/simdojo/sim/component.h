// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file component.h
/// @brief Simulation compound graph: Node, Component, CompositeComponent, Port, Link, and
/// QueuedLink.

#ifndef SIMDOJO_SIM_COMPONENT_H_
#define SIMDOJO_SIM_COMPONENT_H_

#include "simdojo/sim/event_queue.h"
#include "simdojo/sim/exec_mode.h"
#include "simdojo/sim/message.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace simdojo {

/// @brief Direction of a port (input or output).
enum class PortDirection : uint8_t {
  IN,  ///< Receives messages.
  OUT, ///< Sends messages.
};

/// @brief Protocol tag for type-safe port wiring.
///
/// @details Links can only be created between ports with the same protocol.
/// This prevents nonsensical connections (e.g., wiring a memory port to a
/// dispatch port). UNTYPED is the default and allows any connection.
/// Both requester and completer sides of the same protocol use the same tag;
/// direction (IN/OUT) distinguishes the role.
enum class PortProtocol : uint8_t {
  UNTYPED,  ///< No protocol constraint (default).
  MEMORY,   ///< Memory transactions (load/store/atomic).
  DISPATCH, ///< Workgroup/wavefront dispatch.
  STATUS,   ///< Status reporting (idle, busy, done).
};

class Port;
class Link;
class SimulationEngine;

/// @brief Base node in the simulation compound graph.
///
/// @details Nodes form a compound graph where edges are either:
///   - Inclusion edges (parent-child in the component tree)
///   - Adjacency edges (links between ports)
class Node {
public:
  /// @brief Construct a node with the given name.
  /// @param[in] name Human-readable name for this node.
  explicit Node(std::string name) : name_(std::move(name)) {}
  virtual ~Node() = default;

  /// @brief Return the human-readable name.
  /// @returns Const reference to the name string.
  const std::string &name() const { return name_; }

  /// @brief Return the auto-assigned unique identifier.
  /// @returns The ComponentID.
  ComponentID id() const { return id_; }

  /// @brief Return the parent node in the component tree.
  /// @returns Pointer to the parent, or nullptr if this is the root.
  Node *parent() const { return parent_; }

  /// @brief Set the parent node.
  /// @param[in] parent Pointer to the new parent node.
  void set_parent(Node *parent) { parent_ = parent; }

  /// @brief Return the depth in the component tree (root = 0).
  /// @returns Depth value.
  uint32_t depth() const { return depth_; }

  /// @brief Set the depth in the component tree.
  /// @param[in] d New depth value.
  void set_depth(uint32_t d) { depth_ = d; }

  /// @brief Full hierarchical path name (e.g., "soc.subsystem0.unit3").
  /// @returns Dot-separated path from the root to this node.
  std::string full_path() const {
    if (!parent_)
      return name_;
    return parent_->full_path() + "." + name_;
  }

private:
  /// @brief Global counter for auto-assigning unique IDs.
  static inline std::atomic<ComponentID> next_id_ = 0;

  const std::string name_; ///< Human-readable node name.
  const ComponentID id_ =
      next_id_.fetch_add(1, std::memory_order_relaxed); ///< Auto-assigned unique identifier.
  Node *parent_ = nullptr;                              ///< Parent in the component tree.
  uint32_t depth_ = 0;                                  ///< Depth in the component tree.
};

/// @brief A component in the simulation compound graph.
///
/// @details Components are the active simulation entities. They own ports
/// and interact with the simulation engine. Event handlers are registered
/// directly on Ports via Port::set_handler().
class Component : public Node {
public:
  /// @brief Construct a component with the given name.
  /// @param[in] name Human-readable name for this component.
  explicit Component(std::string name) : Node(std::move(name)) {}
  ~Component() override = default;

  /// @brief Called once before simulation starts.
  virtual void initialize() {}

  /// @brief Called after initialize to kick off steady-state execution.
  ///
  /// @details Components override this to schedule their initial events (for
  /// event-driven mode) or prepare for their run() loop (functional mode).
  /// Called on all components before the main simulation loop starts.
  virtual void startup() {}

  /// @brief Called once after simulation ends. Override to release resources.
  virtual void shutdown() {}

  /// @brief Execute one unit of component-specific work.
  ///
  /// What constitutes a "step" is defined by each component: one instruction
  /// for a compute unit, one packet for a command processor, etc. The engine
  /// does not call this — it is a domain-level API for use on concrete types.
  /// @retval true Component has more work.
  /// @retval false Component is idle.
  virtual bool step() { return false; }

  /// @brief Return the list of ports owned by this component.
  /// @returns Const reference to the port vector.
  const std::vector<std::unique_ptr<Port>> &ports() const { return ports_; }

  /// @brief Add a port to this component. Ownership is transferred.
  /// @param[in] port The port to add (ownership transferred).
  /// @returns Raw pointer to the added port.
  Port *add_port(std::unique_ptr<Port> port);

  /// @brief Find a port by its ID.
  /// @param[in] port_id The port ID to search for.
  /// @returns Pointer to the port, or nullptr if not found.
  Port *find_port(PortID port_id) const;

  /// @brief Return the partition this component is assigned to.
  /// @returns The partition ID, or INVALID_PARTITION_ID if unassigned.
  PartitionID partition_id() const { return partition_id_; }

  /// @brief Assign this component to a partition.
  /// @param[in] pid The partition ID to assign.
  void set_partition_id(PartitionID pid) { partition_id_ = pid; }

  /// @brief Return the partitioning weight.
  /// @returns The component's weight.
  uint32_t weight() const { return weight_; }

  /// @brief Set the partitioning weight.
  /// @param[in] w New weight value.
  void set_weight(uint32_t w) { weight_ = w; }

  /// @brief Whether this component is a composite (has children).
  /// @retval false This is a leaf component.
  virtual bool is_composite() const { return false; }

  /// @brief Return the simulation engine this component belongs to.
  /// @returns Pointer to the engine, or nullptr if not yet registered.
  SimulationEngine *engine() const { return engine_; }

  /// @brief Associate this component with a simulation engine.
  /// @param[in] e Pointer to the simulation engine.
  void set_engine(SimulationEngine *e) { engine_ = e; }

protected:
  /// @brief Schedule an event into this component's partition queue.
  ///
  /// @details Convenience method for subclasses. Delegates to the engine's
  /// private schedule_event(). Must only be called from the owning partition's
  /// thread during event processing.
  void schedule_event(Event *event, Tick timestamp, std::unique_ptr<Message> message = nullptr);

  /// @brief The current simulation tick of the partition that owns this
  ///        component.
  ///
  /// @details What a component needs to know whether a tick it has been handed
  /// is in the past. Only meaningful while the owning partition is processing
  /// events; before the engine starts it is zero.
  /// @returns The owning partition's current tick.
  Tick current_tick() const;

  /// @brief Whether this component is owned by a live partition context.
  ///
  /// @details A non-null engine pointer is not enough on its own: shutdown()
  /// destroys the partition contexts and leaves every component holding its
  /// engine, and a component no partition claimed keeps INVALID_PARTITION_ID.
  /// @retval true The owning partition context exists and can be indexed.
  /// @retval false This component has no live partition to schedule into.
  bool attached() const;

  /// @brief Schedule a collapsing wake for @p event.
  ///
  /// @details Convenience method for subclasses. Delegates to
  /// SimulationEngine::schedule_wake(): the event keeps at most one
  /// outstanding entry, an earlier ask supersedes a later one, and a wake into
  /// the past is clamped forward rather than refused. Same owner-thread-only
  /// contract as schedule_event().
  /// @param event Event to wake.
  /// @param timestamp Requested wake tick.
  /// @retval true The wake was armed, superseding any pending one.
  /// @retval false An earlier or equal wake was already pending.
  bool schedule_wake(Event *event, Tick timestamp);

  /// @brief Whether @p event still has a wake outstanding.
  /// @param event Event to query.
  /// @retval true A wake is queued for it in this create() generation.
  /// @retval false Nothing is queued for it.
  bool wake_pending(const Event &event) const;

private:
  std::vector<std::unique_ptr<Port>> ports_;        ///< Owned ports.
  PartitionID partition_id_ = INVALID_PARTITION_ID; ///< Assigned partition.
  uint32_t weight_ = 1;                             ///< Partitioning weight.
  SimulationEngine *engine_ = nullptr;              ///< Owning engine.
};

/// @brief A component that contains child components (inclusion edges).
class CompositeComponent : public Component {
public:
  /// @brief Construct a composite component with the given name.
  /// @param[in] name Human-readable name.
  explicit CompositeComponent(std::string name) : Component(std::move(name)) {}
  ~CompositeComponent() override = default;

  /// @brief This is a composite component.
  bool is_composite() const override { return true; }

  /// @brief Add a child component. Sets the child's parent and depth.
  /// @param[in] child The child to add (ownership transferred).
  /// @returns Raw pointer to the added child.
  Component *add_child(std::unique_ptr<Component> child);

  /// @brief Return the list of child components.
  /// @returns Const reference to the children vector.
  const std::vector<std::unique_ptr<Component>> &children() const { return children_; }

  /// @brief Find a direct child by name.
  /// @param[in] name The name to search for.
  /// @returns Pointer to the child, or nullptr if not found.
  Component *find_child(const std::string &name) const;

  /// @brief Move all children from another composite into this one.
  /// @details Used to absorb a component's subtree without changing paths.
  void adopt_children(CompositeComponent &donor);

  /// @brief Recursively collect all components in the subtree, including this composite.
  /// @param[out] out Vector to append components into.
  void collect_components(std::vector<Component *> &out);

  /// @brief Return the total number of descendants (recursive).
  /// @returns Count of all children, grandchildren, etc.
  uint32_t num_descendants() const;

private:
  std::vector<std::unique_ptr<Component>> children_; ///< Owned child components.
};

/// @brief A directional connection between two ports.
///
/// @details Links carry messages between ports and model communication latency.
/// They are the adjacency edges in the compound graph.
class Link {
public:
  /// @brief Construct a link between two ports.
  /// @param[in] id Unique link identifier.
  /// @param[in] src Source port.
  /// @param[in] dst Destination port.
  /// @param[in] latency Propagation delay in simulation ticks.
  Link(LinkID id, Port *src, Port *dst, Tick latency)
      : id_(id), src_(src), dst_(dst), latency_(latency) {}
  virtual ~Link() = default;

  /// @brief Return the unique link identifier.
  /// @returns The LinkID.
  LinkID id() const { return id_; }

  /// @brief Return the source port.
  /// @returns Pointer to the source port.
  Port *src() const { return src_; }

  /// @brief Return the destination port.
  /// @returns Pointer to the destination port.
  Port *dst() const { return dst_; }

  /// @brief Return the propagation latency in simulation ticks.
  /// @returns Latency value.
  Tick latency() const { return latency_; }

  /// @brief Send a message departing now.
  ///
  /// @details Routes to the local queue or a cross-partition inbox according
  /// to partition assignment.
  /// @param[in] msg The message to send (ownership transferred).
  /// @note Not virtual: send_at() is the single customization hook, and this
  ///       is a fixed wrapper that names "now" as the departure. A subclass
  ///       declaring its own send() shadows rather than overrides it, and
  ///       every caller holding a Link* would keep reaching this one.
  /// @throws std::logic_error if a clocked sender is not attached to a live
  ///         engine. A functional link needs no engine and needs no tick.
  void send(std::unique_ptr<Message> msg) {
    // Read into a local first: the departure and the moved-from message are
    // two arguments of one call with unspecified evaluation order, so this is
    // what keeps who owns the message on a refusal off the compiler.
    const Tick ready_tick = depart_now_or_zero();
    send_at(std::move(msg), ready_tick);
  }

  /// @brief Send a message that the sender makes ready at @p ready_tick.
  ///
  /// @details The departure tick is a parameter rather than something read
  /// back out of the message, so it means exactly what the caller said and
  /// nothing the message happens to be carrying can change it. A message being
  /// forwarded still holds the departure stamp of the hop it arrived on; that
  /// stamp is overwritten here rather than treated as a request.
  /// @param[in] msg The message to send (ownership transferred).
  /// @param[in] ready_tick Tick the sender makes the message ready.
  /// @throws std::logic_error if a clocked sender is not attached to a live
  ///         engine.
  /// @throws std::invalid_argument if @p ready_tick is before a clocked
  ///         sender's current tick, or is TICK_MAX, or if the arrival that
  ///         follows from it saturates. A functional link has no clock and
  ///         throws neither.
  virtual void send_at(std::unique_ptr<Message> msg, Tick ready_tick);

  /// @brief Whether this link crosses a partition boundary.
  /// @retval true Source and destination are in different partitions.
  /// @retval false Both endpoints are in the same partition.
  bool is_cross_partition() const;

  /// @brief Return the link weight (for partitioning cut cost).
  /// @returns The weight value.
  uint32_t weight() const { return weight_; }

  /// @brief Set the link weight.
  /// @param[in] w New weight value.
  void set_weight(uint32_t w) { weight_ = w; }

  /// @brief Return the execution mode for this link.
  /// @returns The ExecMode (FUNCTIONAL or CLOCKED).
  ExecMode exec_mode() const { return exec_mode_; }

  /// @brief Set the execution mode for this link.
  ///
  /// @details In FUNCTIONAL mode, send() directly invokes the destination
  /// port's handler synchronously (zero event overhead). In CLOCKED mode,
  /// send() routes through the simulation engine's event queue with
  /// propagation latency.
  /// @param[in] mode The execution mode to set.
  void set_exec_mode(ExecMode mode) { exec_mode_ = mode; }

protected:
  /// @brief The current tick of the partition this link sends from.
  ///
  /// @details The partition's local processing tick, not GVT: GVT is stale in
  /// multi-threaded mode, which would let a cross-partition message arrive
  /// before what its neighbours have already processed.
  /// @returns The sender's current tick.
  /// @throws std::logic_error if the sender is not attached to a live engine.
  Tick depart_now() const;

  /// @brief The tick a send-now departs at, on a link of either mode.
  ///
  /// @details Tick zero on a functional link, which has no simulated time and
  /// no engine to ask for one, and the sender's current tick on a clocked one.
  /// @returns The departure tick for a send that names none.
  /// @throws std::logic_error if a clocked sender is not attached to a live
  ///         engine.
  Tick depart_now_or_zero() const;

  /// @brief Stamp @p message to depart at @p ready_tick and return its arrival.
  ///
  /// @details Every clocked send goes through here, so a message's departure
  /// tick, its propagation latency, and the arrival that follows from them are
  /// decided in one place.
  ///
  /// A departure already in the past is refused rather than clamped. It would
  /// otherwise be scheduled behind the current tick, and the engine, being a
  /// min-heap with no floor, would pop it next and move simulated time
  /// backwards. TICK_MAX is refused too: it is the "no such tick" value, and a
  /// saturated completion tick means the sender computed a deadline it cannot
  /// meet rather than one at the end of time. An arrival that saturates to
  /// TICK_MAX is refused for the same reason: it is the value an empty queue
  /// reports, so such a message would be invisible to every scheduler that
  /// asks a queue when it next has work.
  /// @param message Message to stamp.
  /// @param ready_tick Tick the sender makes the message ready.
  /// @returns The tick the message arrives at the destination port.
  /// @throws std::invalid_argument if @p ready_tick, or the arrival that
  ///         follows from it, is unusable.
  Tick stamp_for_send(Message &message, Tick ready_tick) const;

private:
  LinkID id_;                              ///< Unique link identifier.
  Port *src_;                              ///< Source port endpoint.
  Port *dst_;                              ///< Destination port endpoint.
  Tick latency_;                           ///< Propagation delay in ticks.
  uint32_t weight_ = 1;                    ///< Partitioning cut weight.
  ExecMode exec_mode_ = ExecMode::CLOCKED; ///< Execution mode (default: event-based).
};

/// @brief A named connection point on a Component.
///
/// @details Ports are the endpoints of Links. Each port belongs to a single component
/// and connects to at most one peer port via a link.
class Port : public Node {
public:
  /// @brief Construct a port.
  /// @param[in] name Human-readable port name.
  /// @param[in] port_id Unique port identifier within the component.
  /// @param[in] owner The component that owns this port.
  /// @param[in] direction Whether this port sends or receives messages.
  /// @param[in] protocol Protocol tag for type-safe wiring (default: UNTYPED).
  Port(std::string name, PortID port_id, Component *owner,
       PortDirection direction = PortDirection::OUT, PortProtocol protocol = PortProtocol::UNTYPED)
      : Node(std::move(name)), port_id_(port_id), owner_(owner), direction_(direction),
        protocol_(protocol) {
    set_parent(owner);
  }

  /// @brief Return the port identifier.
  /// @returns The PortID.
  PortID port_id() const { return port_id_; }

  /// @brief Return the component that owns this port.
  /// @returns Pointer to the owning component.
  Component *owner() const { return owner_; }

  /// @brief Return the link attached to this port.
  /// @returns Pointer to the link, or nullptr if unconnected.
  Link *link() const { return link_; }

  /// @brief Attach a link to this port.
  /// @param[in] lnk Pointer to the link to attach.
  void set_link(Link *lnk) { link_ = lnk; }

  /// @brief The peer port at the other end of the link.
  /// @returns Pointer to the peer port, or nullptr if unconnected.
  Port *peer() const {
    if (link_ == nullptr)
      return nullptr;
    return (link_->src() == this) ? link_->dst() : link_->src();
  }

  /// @brief Send a message through this port's link.
  /// @param[in] msg The message to send (ownership transferred).
  void send(std::unique_ptr<Message> msg) {
    address(*msg);
    link_->send(std::move(msg));
  }

  /// @brief Send a message that becomes ready at @p ready_tick.
  ///
  /// @details The message departs at @p ready_tick rather than now, so it
  /// arrives @p ready_tick plus the link's latency later. A clocked server
  /// that has reserved itself until a completion tick uses this to answer at
  /// that tick without scheduling an event to wake itself up and do it, which
  /// is the difference between one event per request and two.
  ///
  /// The link's own propagation is added on top, not folded in: @p ready_tick
  /// is when the *sender* is finished, and the crossing costs what the link
  /// says it costs.
  ///
  /// What a functional link does with @p ready_tick depends on how it carries
  /// messages. One that calls the destination handler synchronously has no
  /// notion of when and ignores it; a buffered one still stamps it, because a
  /// buffer has to order and release what it holds. A functional link has no
  /// clock either way, so @p ready_tick is never compared against a current
  /// tick there and cannot be refused for being in the past.
  /// @param[in] msg The message to send (ownership transferred).
  /// @param[in] ready_tick Tick at which the sender makes the message ready;
  ///            must not be before the sender's current tick, and must not be
  ///            TICK_MAX.
  /// @throws std::invalid_argument if @p ready_tick is unusable.
  /// @throws std::logic_error if the sender is not attached to a live engine.
  void send_at(std::unique_ptr<Message> msg, Tick ready_tick) {
    address(*msg);
    link_->send_at(std::move(msg), ready_tick);
  }

  /// @brief Return the port's reusable message-arrival event.
  /// @returns Pointer to the event.
  Event *recv_event() { return &recv_event_; }

  /// @brief Set the handler invoked when a message arrives at this port.
  /// @param h Handler callback.
  void set_handler(EventHandler h) { recv_event_.set_handler(std::move(h)); }

  /// @brief Return the port direction (IN or OUT).
  /// @returns The PortDirection.
  PortDirection direction() const { return direction_; }

  /// @brief Return the port protocol tag.
  /// @returns The PortProtocol.
  PortProtocol protocol() const { return protocol_; }

private:
  /// @brief Check this port can send and stamp @p msg with both endpoints.
  ///
  /// @details Shared by send() and send_at() so a precondition added to one
  /// path cannot go missing from the other.
  /// @param[in,out] msg Message to address.
  void address(Message &msg) {
    assert(link_ != nullptr && "Port::send called on unconnected port");
    assert(direction_ == PortDirection::OUT && "can only send from OUT ports");
    Port *p = peer();
    assert(p != nullptr && "Port::send peer is null");
    msg.set_ports(port_id_, p->port_id());
  }

  PortID port_id_;          ///< Port identifier within the component.
  Component *owner_;        ///< Owning component.
  PortDirection direction_; ///< Input or output.
  PortProtocol protocol_;   ///< Protocol tag for type-safe wiring.
  Link *link_ = nullptr;    ///< Attached link (nullptr if unconnected).
  Event recv_event_{owner_, EventType::MESSAGE_ARRIVAL}; ///< Reusable event for message arrivals.
};

/// @brief A link that buffers messages in a timestamp-ordered queue.
///
/// @details Unlike the base Link which immediately routes messages through the
/// simulation engine, QueuedLink stores them internally. The receiving
/// component explicitly pops or drains messages when ready.
class QueuedLink : public Link {
public:
  /// @brief Construct a queued link with bounded capacity.
  /// @param[in] id Unique link identifier.
  /// @param[in] src Source port.
  /// @param[in] dst Destination port.
  /// @param[in] latency Propagation delay in simulation ticks.
  /// @param[in] capacity Maximum number of buffered messages.
  QueuedLink(LinkID id, Port *src, Port *dst, Tick latency, size_t capacity)
      : Link(id, src, dst, latency), queue_(capacity) {}

  /// @brief Enqueue a message departing now, without asserting on a full queue.
  /// @param[in] msg The message to enqueue (ownership transferred).
  /// @retval true Message was enqueued successfully.
  /// @retval false Queue is full; the message was destroyed.
  /// @throws Whatever send_at() throws.
  bool try_send(std::unique_ptr<Message> msg) {
    const Tick ready_tick = depart_now_or_zero();
    return try_send_at(std::move(msg), ready_tick);
  }

  /// @brief Enqueue a message departing at @p ready_tick, without asserting on
  ///        a full queue.
  /// @param[in] msg The message to enqueue (ownership transferred).
  /// @param[in] ready_tick Tick the sender makes the message ready.
  /// @retval true Message was enqueued successfully.
  /// @retval false Queue is full; the message was destroyed.
  /// @throws Whatever send_at() throws.
  bool try_send_at(std::unique_ptr<Message> msg, Tick ready_tick) {
    stamp_for_send(*msg, ready_tick);
    return queue_.push(std::move(msg));
  }

  /// @brief Enqueue a message, asserting the queue is not full.
  ///
  /// @details A full queue is a modelling error on this entry point rather
  /// than backpressure; use try_send_at() to be told instead. Note the assert
  /// is compiled out under NDEBUG, where an overflowing message is destroyed
  /// silently -- callers that can overflow should use try_send_at().
  /// @param[in] msg The message to enqueue (ownership transferred).
  /// @param[in] ready_tick Tick the sender makes the message ready.
  void send_at(std::unique_ptr<Message> msg, Tick ready_tick) override {
    [[maybe_unused]] bool ok = try_send_at(std::move(msg), ready_tick);
    assert(ok && "QueuedLink: send on full queue");
  }

  /// @brief Pop the next message from the queue (asserts non-empty).
  /// @returns The oldest message.
  std::unique_ptr<Message> pop() { return queue_.pop(); }

  /// @brief Peek at the next message without removing it.
  /// @returns Pointer to the oldest message, or nullptr if empty.
  const Message *peek() const { return queue_.peek(); }

  /// @brief Check whether the queue is empty.
  /// @retval true No messages are buffered.
  /// @retval false At least one message is buffered.
  bool empty() const { return queue_.empty(); }

  /// @brief Check whether the queue is at capacity.
  /// @retval true Queue has reached its maximum capacity.
  /// @retval false Queue has room for more messages.
  bool full() const { return queue_.full(); }

  /// @brief Return the number of buffered messages.
  /// @returns Current queue size.
  size_t size() const { return queue_.size(); }

  /// @brief Return the maximum number of messages the queue can hold.
  /// @returns Queue capacity.
  size_t capacity() const { return queue_.capacity(); }

  /// @brief Return the arrival tick of the next message.
  /// @returns Tick of the oldest message, or TICK_MAX if empty.
  Tick next_message_time() const { return queue_.next_message_time(); }

  /// @brief Pop all messages with arrival_tick <= current_time.
  /// @param[in] current_time The current simulation tick.
  /// @param[out] out Vector to append ready messages into.
  void drain_ready(Tick current_time, std::vector<std::unique_ptr<Message>> &out) {
    while (!queue_.empty() && queue_.next_message_time() <= current_time) {
      out.push_back(queue_.pop());
    }
  }

private:
  MessageQueue queue_; ///< Bounded timestamp-ordered message buffer.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_COMPONENT_H_
