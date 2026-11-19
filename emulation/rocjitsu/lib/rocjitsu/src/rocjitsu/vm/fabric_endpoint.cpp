// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/fabric_endpoint.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"
#include "util/except.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {
using amdgpu::VmAccessOutcome;

class FabricEndpoint::Request {
public:
  enum class EffectPhase : uint8_t { Queued, Entered, Committing, Done, Revoked };

  void cancel_queued() noexcept {
    EffectPhase expected = EffectPhase::Queued;
    phase.compare_exchange_strong(expected, EffectPhase::Revoked, std::memory_order_acq_rel);
    cancelled.store(true, std::memory_order_release);
  }

  void cancel_before_effect() noexcept {
    EffectPhase expected = phase.load(std::memory_order_acquire);
    while (
        (expected == EffectPhase::Queued || expected == EffectPhase::Entered) &&
        !phase.compare_exchange_weak(expected, EffectPhase::Revoked, std::memory_order_acq_rel)) {
    }
    cancelled.store(true, std::memory_order_release);
  }

  uint64_t id = 0;
  uint64_t generation = 0;
  size_t route_index = 0;
  size_t source_position = 0;
  size_t target_position = 0;
  int step = 1;
  bool write = false;
  bool local = false;
  TraceOrigin trace_origin{};
  FabricBacking backing;
  std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate;
  EffectAdmission admit_effect;
  PreEffectAdmission admit_pre_effect;
  amdgpu::MemoryIngressPermit request_reservation;
  // Protected by State::mutex. A retry may be Queued while its prior target
  // handler still uses request_reservation.
  bool active_handler = false;
  std::vector<std::byte> payload;
  std::vector<std::byte> response;
  std::vector<std::byte> effect_response;
  std::optional<CanonicalLineFreshness::ReadStamp> read_stamp;
  std::mutex mutex;
  VmAccessOutcome outcome = VmAccessOutcome::Unavailable;
  VmAccessOutcome effect_outcome = VmAccessOutcome::Unavailable;
  // The terminal effect result and read bytes are published before the VM
  // effect permit is released. Revocation can therefore distinguish a queued
  // request from a committed effect without waiting for the clocked reply.
  std::atomic<EffectPhase> phase{EffectPhase::Queued};
  std::atomic<bool> cancelled{false};
};

class FabricEndpoint::State {
public:
  std::mutex mutex;
  FabricEndpoint *owner = nullptr;
  bool accepting = false;
  uint64_t generation = 0;
  uint64_t next_id = 0;
  std::deque<std::shared_ptr<Request>> inbox;
  std::unordered_map<uint64_t, std::shared_ptr<Request>> pending;
  size_t active_pre_admissions = 0;
  size_t active_effects = 0;
  size_t active_reservation_releases = 0;
  std::condition_variable effects_drained;
  Diagnostics diagnostics;
  size_t effect_trace_capacity = 0;

  // Call with mutex held. The caller releases the detached permit outside the
  // mutex, then calls finish_reservation_release() before it can return.
  amdgpu::MemoryIngressPermit take_idle_reservation(Request &request) {
    if (request.active_handler)
      return {};
    amdgpu::MemoryIngressPermit reservation = std::move(request.request_reservation);
    if (reservation)
      ++active_reservation_releases;
    return reservation;
  }

  void finish_reservation_release(amdgpu::MemoryIngressPermit &reservation) {
    if (!reservation)
      return;
    reservation.reset();
    std::lock_guard lock(mutex);
    --active_reservation_releases;
    effects_drained.notify_all();
  }
};

class FabricEndpoint::WireMessage final : public simdojo::Message {
public:
  WireMessage(uint64_t id, size_t route, size_t position, int step, bool response, bool write,
              VmAccessOutcome outcome, std::vector<std::byte> bytes)
      : id(id), route(route), position(position), step(step), response(response), write(write),
        outcome(outcome), bytes(std::move(bytes)) {
    header().sequence_num = id;
    header().op = response ? simdojo::MessageOp::RESPONSE
                           : (write ? simdojo::MessageOp::WRITE : simdojo::MessageOp::READ);
    header().size_bytes = static_cast<uint32_t>(this->bytes.size());
  }

  uint64_t id;
  size_t route;
  size_t position;
  int step;
  bool response;
  bool write;
  VmAccessOutcome outcome;
  std::vector<std::byte> bytes;
};

class FabricEndpoint::Node final : public simdojo::Component {
public:
  Node(FabricEndpoint &fabric, std::string endpoint_id, size_t ordinal)
      : Component("endpoint" + std::to_string(ordinal)), fabric_(fabric),
        endpoint_id_(std::move(endpoint_id)) {
    set_weight(0);
  }

  struct EdgePorts {
    simdojo::Port *out = nullptr;
    simdojo::Port *in = nullptr;
  };

  EdgePorts add_edge(const std::string &neighbor) {
    const auto found = edges_.find(neighbor);
    if (found != edges_.end())
      return found->second;
    const auto ordinal = edges_.size();
    const auto out_id = static_cast<simdojo::PortID>(ports().size());
    auto *out = add_port(std::make_unique<simdojo::Port>("out" + std::to_string(ordinal), out_id,
                                                         this, simdojo::PortDirection::OUT,
                                                         simdojo::PortProtocol::MEMORY));
    const auto in_id = static_cast<simdojo::PortID>(ports().size());
    auto *in = add_port(std::make_unique<simdojo::Port>("in" + std::to_string(ordinal), in_id, this,
                                                        simdojo::PortDirection::IN,
                                                        simdojo::PortProtocol::MEMORY));
    in->set_handler([this](simdojo::Tick, simdojo::Message *message) {
      if (auto *wire = dynamic_cast<WireMessage *>(message))
        fabric_.receive(*this, *wire);
    });
    return edges_.emplace(neighbor, EdgePorts{out, in}).first->second;
  }

  simdojo::Port *out_to(const std::string &neighbor) const {
    const auto found = edges_.find(neighbor);
    return found == edges_.end() ? nullptr : found->second.out;
  }

  const std::string &endpoint_id() const { return endpoint_id_; }

private:
  FabricEndpoint &fabric_;
  std::string endpoint_id_;
  std::unordered_map<std::string, EdgePorts> edges_;
};

struct FabricEndpoint::Route {
  uint32_t source_gpu_id = 0;
  uint32_t target_gpu_id = 0;
  std::vector<Node *> nodes;
};

class FabricEndpoint::Operation final : public amdgpu::VmDeferredTransfer {
public:
  Operation(std::shared_ptr<State> state, std::shared_ptr<Request> request)
      : state_(std::move(state)), request_(std::move(request)) {}

  void submit() override {
    if (submitted_.exchange(true, std::memory_order_acq_rel))
      return;
    std::lock_guard lock(state_->mutex);
    if (!state_->accepting || state_->generation != request_->generation ||
        state_->owner == nullptr || request_->cancelled.load(std::memory_order_acquire)) {
      request_->cancel_queued();
      std::lock_guard request_lock(request_->mutex);
      request_->outcome = VmAccessOutcome::Revoked;
      return;
    }
    state_->pending.emplace(request_->id, request_);
    state_->inbox.push_back(request_);
    ++state_->diagnostics.submissions;
    // This call only queues a reusable event. Link::send is reached later by
    // service_inbox() on the simulation owner's partition thread.
    state_->owner->engine()->schedule_event_now(&state_->owner->inbox_event_);
  }

  VmAccessOutcome poll(std::span<std::byte> result) override {
    std::lock_guard lock(request_->mutex);
    if (request_->outcome == VmAccessOutcome::Unavailable &&
        request_->cancelled.load(std::memory_order_acquire)) {
      const Request::EffectPhase phase = request_->phase.load(std::memory_order_acquire);
      if (phase == Request::EffectPhase::Queued || phase == Request::EffectPhase::Entered ||
          phase == Request::EffectPhase::Committing)
        return VmAccessOutcome::Unavailable;
      request_->outcome =
          phase == Request::EffectPhase::Done ? request_->effect_outcome : VmAccessOutcome::Revoked;
      if (phase == Request::EffectPhase::Done && !request_->write &&
          request_->effect_outcome == VmAccessOutcome::Complete)
        request_->response = request_->effect_response;
    }
    if (request_->outcome == VmAccessOutcome::Complete && !request_->write) {
      if (result.size() < request_->response.size())
        return VmAccessOutcome::Malformed;
      std::copy(request_->response.begin(), request_->response.end(), result.begin());
    }
    return request_->outcome;
  }

  VmAccessOutcome reconcile_revocation(std::span<std::byte> result) override {
    Request::EffectPhase phase = request_->phase.load(std::memory_order_acquire);
    if (phase == Request::EffectPhase::Queued) {
      Request::EffectPhase expected = Request::EffectPhase::Queued;
      if (request_->phase.compare_exchange_strong(expected, Request::EffectPhase::Revoked,
                                                  std::memory_order_acq_rel)) {
        request_->cancelled.store(true, std::memory_order_release);
        release_reservation_if_idle();
        return VmAccessOutcome::Revoked;
      }
      phase = expected;
    }
    if (phase == Request::EffectPhase::Revoked) {
      release_reservation_if_idle();
      return VmAccessOutcome::Revoked;
    }
    if (phase == Request::EffectPhase::Entered || phase == Request::EffectPhase::Committing)
      return VmAccessOutcome::Unavailable;
    if (request_->effect_outcome == VmAccessOutcome::Complete && !request_->write) {
      if (result.size() < request_->effect_response.size())
        return VmAccessOutcome::Malformed;
      std::copy(request_->effect_response.begin(), request_->effect_response.end(), result.begin());
    }
    return request_->effect_outcome;
  }

  std::optional<CanonicalLineFreshness::ReadStamp> completed_read_stamp() const override {
    if (request_->phase.load(std::memory_order_acquire) != Request::EffectPhase::Done ||
        request_->write || request_->effect_outcome != VmAccessOutcome::Complete)
      return std::nullopt;
    return request_->read_stamp;
  }

  void cancel() noexcept override {
    request_->cancel_before_effect();
    release_reservation_if_idle();
  }

private:
  void release_reservation_if_idle() noexcept {
    amdgpu::MemoryIngressPermit reservation;
    {
      std::lock_guard lock(state_->mutex);
      reservation = state_->take_idle_reservation(*request_);
    }
    state_->finish_reservation_release(reservation);
  }

  std::shared_ptr<State> state_;
  std::shared_ptr<Request> request_;
  std::atomic<bool> submitted_{false};
};

FabricEndpoint::FabricEndpoint(
    std::vector<config::SystemPeerRoute> routes,
    std::vector<std::shared_ptr<amdgpu::DeviceCacheCoherence>> coherence_domains)
    : CompositeComponent("fabric"), routes_(std::move(routes)),
      coherence_domains_(std::move(coherence_domains)), state_(std::make_shared<State>()) {
  std::unordered_set<const amdgpu::DeviceCacheCoherence *> seen_domains;
  for (const std::shared_ptr<amdgpu::DeviceCacheCoherence> &domain : coherence_domains_)
    if (!domain || !seen_domains.insert(domain.get()).second)
      throw util::ConfigError("fabric coherence domains must be non-null and distinct");
  set_weight(0);
  inbox_event_.set_handler([this](simdojo::Tick, simdojo::Message *) { service_inbox(); });
}

FabricEndpoint::~FabricEndpoint() { shutdown(); }

void FabricEndpoint::configure_cache_writeback(std::vector<amdgpu::DeviceCacheDomain> domains,
                                               amdgpu::PreparedWritebackResolver resolver) {
  if (wired_ || !resolver || domains.size() != coherence_domains_.size())
    throw util::ConfigError("fabric cache writeback domains are incomplete");
  std::unordered_set<uint32_t> gpu_ids;
  // A legacy single-GPU KFD receives its public GPU ID in setup_topology(),
  // after topology assembly. Resolve that one domain from the admitted
  // backing's owner when handling an effect.
  const bool deferred_owner_id = domains.size() == 1 && routes_.empty();
  for (size_t i = 0; i < domains.size(); ++i) {
    if (domains[i].coherence != coherence_domains_[i] ||
        (domains[i].gpu_id == 0 && !deferred_owner_id) || !gpu_ids.insert(domains[i].gpu_id).second)
      throw util::ConfigError("fabric cache writeback domain identity is invalid");
  }
  prepared_domains_ = std::move(domains);
  resolve_cache_writeback_ = std::move(resolver);
}

void FabricEndpoint::wire(simdojo::Topology &topology, simdojo::ExecMode mode) {
  if (wired_)
    return;
  wired_ = true;
  std::unordered_map<std::string, Node *> nodes;
  for (const auto &projected : routes_) {
    if (projected.endpoints.size() < 2 ||
        projected.endpoints.size() > config::kMaxPeerRouteEndpoints ||
        projected.source_gpu_id == projected.target_gpu_id)
      continue;
    std::unordered_set<std::string> visited;
    bool valid = true;
    for (const auto &id : projected.endpoints)
      valid &= !id.empty() && visited.insert(id).second;
    if (!valid)
      continue;
    Route route{projected.source_gpu_id, projected.target_gpu_id, {}};
    for (const auto &id : projected.endpoints) {
      auto found = nodes.find(id);
      if (found == nodes.end()) {
        auto child = std::make_unique<Node>(*this, id, nodes.size());
        Node *node = child.get();
        add_child(std::move(child));
        found = nodes.emplace(id, node).first;
      }
      route.nodes.push_back(found->second);
    }
    wired_routes_.push_back(std::move(route));
  }

  // Preallocate one counter per validated route before the engine starts.
  // Recording a committed effect must never allocate after touching backing.
  state_->diagnostics.directed_effects.reserve(wired_routes_.size() * 2);
  for (const Route &route : wired_routes_) {
    state_->diagnostics.directed_effects.push_back(
        {.requester_gpu_id = route.source_gpu_id, .owner_gpu_id = route.target_gpu_id});
    state_->diagnostics.directed_effects.push_back(
        {.requester_gpu_id = route.target_gpu_id, .owner_gpu_id = route.source_gpu_id});
  }

  std::set<std::pair<std::string, std::string>> wired_edges;
  for (const auto &route : wired_routes_) {
    for (size_t i = 1; i < route.nodes.size(); ++i) {
      Node &a = *route.nodes[i - 1];
      Node &b = *route.nodes[i];
      if (!wired_edges.insert(std::minmax(a.endpoint_id(), b.endpoint_id())).second)
        continue;
      const auto a_ports = a.add_edge(b.endpoint_id());
      const auto b_ports = b.add_edge(a.endpoint_id());
      auto [forward, reverse] =
          topology.add_bidirectional_link(a_ports.out, b_ports.in, b_ports.out, a_ports.in, 1, 1);
      forward->set_exec_mode(mode);
      reverse->set_exec_mode(mode);
    }
  }
}

void FabricEndpoint::initialize() {
  std::lock_guard lock(state_->mutex);
  ++state_->generation;
  state_->owner = this;
  state_->accepting = true;
}

void FabricEndpoint::shutdown() {
  std::unordered_map<uint64_t, std::shared_ptr<Request>> pending;
  {
    std::lock_guard lock(state_->mutex);
    state_->accepting = false;
    state_->owner = nullptr;
    ++state_->generation;
    for (const auto &[id, request] : state_->pending) {
      (void)id;
      request->cancel_queued();
    }
    state_->inbox.clear();
    pending.swap(state_->pending);
  }
  for (const auto &[id, request] : pending) {
    (void)id;
    amdgpu::MemoryIngressPermit reservation;
    {
      std::lock_guard lock(state_->mutex);
      reservation = state_->take_idle_reservation(*request);
    }
    state_->finish_reservation_release(reservation);
  }
  {
    std::unique_lock lock(state_->mutex);
    // A target handler must acquire its guard under this mutex before touching
    // backing. Wait for its effects and any concurrent idle lease releases.
    state_->effects_drained.wait(lock, [&] {
      return state_->active_pre_admissions == 0 && state_->active_effects == 0 &&
             state_->active_reservation_releases == 0;
    });
  }
  for (auto &[id, request] : pending) {
    (void)id;
    std::lock_guard lock(request->mutex);
    if (request->outcome == VmAccessOutcome::Unavailable) {
      const Request::EffectPhase phase = request->phase.load(std::memory_order_acquire);
      request->outcome =
          phase == Request::EffectPhase::Done ? request->effect_outcome : VmAccessOutcome::Revoked;
      if (phase == Request::EffectPhase::Done && !request->write &&
          request->effect_outcome == VmAccessOutcome::Complete)
        request->response = request->effect_response;
    }
  }
}

FabricEndpoint::PreparedTransfer FabricEndpoint::prepare_read(
    uint32_t requester_gpu_id, uint32_t owner_gpu_id, FabricBacking backing,
    std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate, EffectAdmission admit_effect,
    PreEffectAdmission admit_pre_effect, TraceOrigin trace_origin) {
  return prepare(requester_gpu_id, owner_gpu_id, std::move(backing), {}, false, false,
                 std::move(effect_gate), std::move(admit_effect), std::move(admit_pre_effect),
                 trace_origin);
}

FabricEndpoint::PreparedTransfer FabricEndpoint::prepare_write(
    uint32_t requester_gpu_id, uint32_t owner_gpu_id, FabricBacking backing,
    std::span<const std::byte> bytes, std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate,
    EffectAdmission admit_effect, PreEffectAdmission admit_pre_effect, TraceOrigin trace_origin) {
  return prepare(requester_gpu_id, owner_gpu_id, std::move(backing), bytes, true, false,
                 std::move(effect_gate), std::move(admit_effect), std::move(admit_pre_effect),
                 trace_origin);
}

FabricEndpoint::PreparedTransfer FabricEndpoint::prepare_local_write(
    uint32_t owner_gpu_id, FabricBacking backing, std::span<const std::byte> bytes,
    std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate, EffectAdmission admit_effect) {
  return prepare(owner_gpu_id, owner_gpu_id, std::move(backing), bytes, true, true,
                 std::move(effect_gate), std::move(admit_effect), {}, {});
}

bool FabricEndpoint::can_route(uint32_t requester_gpu_id, uint32_t owner_gpu_id) const {
  if (requester_gpu_id == owner_gpu_id)
    return false;
  bool found = false;
  for (const Route &route : wired_routes_) {
    const bool match =
        (route.source_gpu_id == requester_gpu_id && route.target_gpu_id == owner_gpu_id) ||
        (route.target_gpu_id == requester_gpu_id && route.source_gpu_id == owner_gpu_id);
    if (match) {
      if (found)
        return false;
      found = true;
    }
  }
  return found;
}

FabricEndpoint::PreparedTransfer
FabricEndpoint::prepare(uint32_t requester_gpu_id, uint32_t owner_gpu_id, FabricBacking backing,
                        std::span<const std::byte> bytes, bool write, bool local,
                        std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate,
                        EffectAdmission admit_effect, PreEffectAdmission admit_pre_effect,
                        TraceOrigin trace_origin) {
  if (!backing || !effect_gate || (local && (owner_gpu_id == 0 || !admit_effect)) ||
      backing.owner_gpu_id() != owner_gpu_id ||
      (local ? (requester_gpu_id != owner_gpu_id || !write) : requester_gpu_id == owner_gpu_id))
    return {};
  if (backing.extent() == 0 || backing.extent() > std::numeric_limits<uint32_t>::max() ||
      (write && bytes.size() != backing.extent()))
    return {.outcome = VmAccessOutcome::Malformed, .operation = {}};

  size_t route_index = wired_routes_.size();
  size_t source_position = 0;
  size_t target_position = 0;
  int step = 1;
  if (!local) {
    for (size_t i = 0; i < wired_routes_.size(); ++i) {
      const Route &route = wired_routes_[i];
      bool forward = route.source_gpu_id == requester_gpu_id && route.target_gpu_id == owner_gpu_id;
      bool reverse = route.target_gpu_id == requester_gpu_id && route.source_gpu_id == owner_gpu_id;
      if (!forward && !reverse)
        continue;
      // Multiple paths for one pair are ambiguous even if their endpoints agree.
      if (route_index != wired_routes_.size())
        return {};
      route_index = i;
      source_position = forward ? 0 : route.nodes.size() - 1;
      target_position = forward ? route.nodes.size() - 1 : 0;
      step = forward ? 1 : -1;
    }
    if (route_index == wired_routes_.size())
      return {};
  }

  auto request = std::make_shared<Request>();
  request->route_index = route_index;
  request->source_position = source_position;
  request->target_position = target_position;
  request->step = step;
  request->write = write;
  request->local = local;
  request->trace_origin = trace_origin;
  request->backing = std::move(backing);
  request->effect_gate = std::move(effect_gate);
  request->admit_effect = std::move(admit_effect);
  request->admit_pre_effect = std::move(admit_pre_effect);
  if (write)
    request->payload.assign(bytes.begin(), bytes.end());
  else
    request->effect_response.resize(request->backing.extent());
  {
    std::lock_guard lock(state_->mutex);
    if (!state_->accepting)
      return {.outcome = VmAccessOutcome::Revoked, .operation = {}};
    request->generation = state_->generation;
    request->id = ++state_->next_id;
    if (request->id == 0)
      request->id = ++state_->next_id;
  }
  return {.outcome = VmAccessOutcome::Complete,
          .operation = std::make_unique<Operation>(state_, std::move(request))};
}

void FabricEndpoint::service_inbox() {
  std::deque<std::shared_ptr<Request>> requests;
  {
    std::lock_guard lock(state_->mutex);
    state_->diagnostics.owner_thread = std::this_thread::get_id();
    requests.swap(state_->inbox);
  }
  for (const auto &request : requests) {
    if (request->cancelled.load(std::memory_order_acquire)) {
      finish(request->id, VmAccessOutcome::Revoked);
      continue;
    }
    if (request->local) {
      {
        std::lock_guard lock(state_->mutex);
        state_->diagnostics.handler_thread = std::this_thread::get_id();
      }
      handle_target(request, nullptr);
      continue;
    }
    const Route &route = wired_routes_[request->route_index];
    WireMessage message(request->id, request->route_index, request->source_position, request->step,
                        false, request->write, VmAccessOutcome::Unavailable, request->payload);
    send_next(route, message);
  }
}

void FabricEndpoint::send_next(const Route &route, WireMessage &message) {
  const auto next = static_cast<int64_t>(message.position) + message.step;
  if (next < 0 || static_cast<size_t>(next) >= route.nodes.size()) {
    finish(message.id, VmAccessOutcome::Faulted);
    return;
  }
  Node *current = route.nodes[message.position];
  Node *neighbor = route.nodes[static_cast<size_t>(next)];
  simdojo::Port *out = current->out_to(neighbor->endpoint_id());
  if (!out || !out->link()) {
    finish(message.id, VmAccessOutcome::Faulted);
    return;
  }
  auto next_message = std::make_unique<WireMessage>(message);
  next_message->position = static_cast<size_t>(next);
  // A received packet carries the preceding link's send tick. Every new hop
  // must be stamped at the current owner tick so clocked latency accumulates.
  next_message->set_timestamp(simdojo::TICK_MAX);
  out->send(std::move(next_message));
}

bool FabricEndpoint::retry_request(const std::shared_ptr<Request> &request) noexcept {
  try {
    std::lock_guard lock(state_->mutex);
    if (!state_->accepting || state_->owner != this || state_->generation != request->generation ||
        !state_->pending.contains(request->id) ||
        request->cancelled.load(std::memory_order_acquire))
      return false;
    state_->inbox.push_back(request);
    Request::EffectPhase expected = Request::EffectPhase::Entered;
    if (!request->phase.compare_exchange_strong(expected, Request::EffectPhase::Queued,
                                                std::memory_order_acq_rel)) {
      state_->inbox.pop_back();
      return false;
    }
    // Waking on the next tick lets a fill blocked behind maintenance finish
    // after its lease is released. Peer requests traverse their route again.
    try {
      engine()->schedule_event_next_tick(&inbox_event_);
    } catch (...) {
      state_->inbox.pop_back();
      expected = Request::EffectPhase::Queued;
      request->phase.compare_exchange_strong(expected, Request::EffectPhase::Revoked,
                                             std::memory_order_acq_rel);
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

void FabricEndpoint::receive(Node &node, WireMessage &message) {
  std::shared_ptr<Request> request;
  {
    std::lock_guard lock(state_->mutex);
    auto found = state_->pending.find(message.id);
    if (found == state_->pending.end())
      return;
    request = found->second;
    state_->diagnostics.handler_thread = std::this_thread::get_id();
    if (message.response)
      ++state_->diagnostics.response_messages;
    else
      ++state_->diagnostics.request_messages;
  }
  if (request->cancelled.load(std::memory_order_acquire)) {
    finish(message.id, VmAccessOutcome::Revoked);
    return;
  }
  if (message.route != request->route_index || message.route >= wired_routes_.size()) {
    finish(message.id, VmAccessOutcome::Faulted);
    return;
  }
  const Route &route = wired_routes_[message.route];
  if (message.position >= route.nodes.size() || route.nodes[message.position] != &node ||
      message.header().completion_status != nullptr || message.payload() != 0 ||
      message.header().addr != 0) {
    finish(message.id, VmAccessOutcome::Faulted);
    return;
  }

  if (message.response) {
    if (message.position == request->source_position) {
      {
        std::lock_guard lock(request->mutex);
        if (request->outcome == VmAccessOutcome::Unavailable) {
          request->outcome = message.outcome;
          if (!request->write && message.outcome == VmAccessOutcome::Complete)
            request->response = message.bytes;
        }
      }
      amdgpu::MemoryIngressPermit reservation;
      {
        std::lock_guard lock(state_->mutex);
        state_->pending.erase(message.id);
        reservation = state_->take_idle_reservation(*request);
      }
      state_->finish_reservation_release(reservation);
      return;
    }
    if (message.step != -request->step) {
      finish(message.id, VmAccessOutcome::Faulted);
      return;
    }
    send_next(route, message);
    return;
  }

  if (message.step != request->step) {
    finish(message.id, VmAccessOutcome::Faulted);
    return;
  }
  if (message.position != request->target_position) {
    send_next(route, message);
    return;
  }

  handle_target(request, &message);
}

void FabricEndpoint::handle_target(const std::shared_ptr<Request> &request,
                                   const WireMessage *message) {
  // Admit the handler under the endpoint mutex before calling frontend state.
  // Shutdown may retire a request after receive() or service_inbox() selects
  // it but before this handler starts. Count the whole handler so shutdown
  // cannot pass its drain in the gap before the effect guard enters.
  class PreAdmissionGuard {
  public:
    PreAdmissionGuard(State &state, const FabricEndpoint *owner, const Request &request)
        : state_(state) {
      std::lock_guard lock(state_.mutex);
      if (state_.accepting && state_.owner == owner && state_.generation == request.generation &&
          state_.pending.contains(request.id)) {
        ++state_.active_pre_admissions;
        active_ = true;
      }
    }
    ~PreAdmissionGuard() {
      if (active_) {
        std::lock_guard lock(state_.mutex);
        --state_.active_pre_admissions;
        state_.effects_drained.notify_all();
      }
    }
    [[nodiscard]] bool active() const { return active_; }

  private:
    State &state_;
    bool active_ = false;
  } pre_admission_guard{*state_, this, *request};
  if (!pre_admission_guard.active())
    return;
  if (request->cancelled.load(std::memory_order_acquire)) {
    finish(request->id, VmAccessOutcome::Revoked);
    return;
  }
  amdgpu::MemoryIngressAdmission pre_effect;
  try {
    if (request->admit_pre_effect)
      pre_effect = request->admit_pre_effect();
  } catch (...) {
    pre_effect.outcome = VmAccessOutcome::Faulted;
  }

  bool admitted = false;
  bool deferred = false;
  {
    std::lock_guard lock(state_->mutex);
    Request::EffectPhase expected = Request::EffectPhase::Queued;
    if (state_->accepting && state_->generation == request->generation &&
        state_->pending.contains(request->id) &&
        !request->cancelled.load(std::memory_order_acquire)) {
      if (request->active_handler) {
        // A retry can reach the target before its prior handler has unwound.
        // Leave its phase Queued and revisit the owner inbox next tick.
        if (request->phase.load(std::memory_order_acquire) == Request::EffectPhase::Queued) {
          try {
            state_->inbox.push_back(request);
            try {
              engine()->schedule_event_next_tick(&inbox_event_);
              deferred = true;
            } catch (...) {
              state_->inbox.pop_back();
            }
          } catch (...) {
          }
        } else {
          // A duplicate target packet must not retire an active effect.
          return;
        }
      } else if (request->phase.compare_exchange_strong(expected, Request::EffectPhase::Entered,
                                                        std::memory_order_acq_rel)) {
        request->active_handler = true;
        ++state_->active_effects;
        admitted = true;
      }
    }
  }
  if (deferred)
    return;
  if (!admitted) {
    finish(request->id, VmAccessOutcome::Revoked);
    return;
  }
  class EffectPermit {
  public:
    EffectPermit(State &state, Request &request, amdgpu::MemoryIngressPermit pre_effect_permit)
        : state_(state), request_(request), pre_effect_permit_(std::move(pre_effect_permit)) {}
    ~EffectPermit() {
      amdgpu::MemoryIngressPermit reservation;
      {
        std::lock_guard lock(state_.mutex);
        const bool retry_pending =
            state_.accepting && state_.owner != nullptr &&
            state_.generation == request_.generation && state_.pending.contains(request_.id) &&
            !request_.cancelled.load(std::memory_order_acquire) &&
            request_.phase.load(std::memory_order_acquire) == Request::EffectPhase::Queued;
        if (!retry_pending)
          reservation = std::move(request_.request_reservation);
        request_.active_handler = false;
      }
      reservation.reset();
      pre_effect_permit_.reset();
      {
        std::lock_guard lock(state_.mutex);
        --state_.active_effects;
        state_.effects_drained.notify_all();
      }
    }

  private:
    State &state_;
    Request &request_;
    amdgpu::MemoryIngressPermit pre_effect_permit_;
  } permit{*state_, *request, std::move(pre_effect.permit)};

  EffectAuthority effect_authority;
  if (pre_effect.outcome != VmAccessOutcome::Complete) {
    effect_authority.outcome = pre_effect.outcome;
  } else {
    try {
      if (request->admit_effect)
        effect_authority =
            request->admit_effect(request->backing, request->write, request->request_reservation);
    } catch (...) {
      effect_authority.outcome = VmAccessOutcome::Faulted;
    }
  }
  const bool authority_ready =
      effect_authority.outcome == VmAccessOutcome::Complete &&
      (!request->admit_effect || static_cast<bool>(effect_authority.permit));

  if (!authority_ready && effect_authority.outcome == VmAccessOutcome::Unavailable) {
    effect_authority.permit.reset();
    if (retry_request(request))
      return;
    Request::EffectPhase expected = Request::EffectPhase::Entered;
    request->phase.compare_exchange_strong(expected, Request::EffectPhase::Revoked,
                                           std::memory_order_acq_rel);
    finish(request->id, VmAccessOutcome::Revoked);
    return;
  }

  if (authority_ready && !coherence_domains_.empty() && !line_freshness_.try_invalidate_all()) {
    effect_authority.permit.reset();
    if (retry_request(request))
      return;
    Request::EffectPhase expected = Request::EffectPhase::Entered;
    request->phase.compare_exchange_strong(expected, Request::EffectPhase::Revoked,
                                           std::memory_order_acq_rel);
    finish(request->id, VmAccessOutcome::Revoked);
    return;
  }

  std::vector<amdgpu::DeviceCacheMaintenanceLease> maintenance;
  const auto release_maintenance = [&] {
    while (!maintenance.empty())
      maintenance.pop_back();
  };
  bool maintenance_ready = authority_ready;
  if (maintenance_ready) {
    try {
      // The legacy single-GPU driver learns its GPU ID after assembly. Its
      // admitted backing supplies the identity for the sole cache domain.
      amdgpu::DeviceCacheDomain owner_domain;
      std::span<const amdgpu::DeviceCacheDomain> effect_domains = prepared_domains_;
      if (prepared_domains_.size() == 1 && prepared_domains_[0].gpu_id == 0) {
        owner_domain = {prepared_domains_[0].coherence, request->backing.owner_gpu_id()};
        effect_domains = {&owner_domain, 1};
      }
      if (!effect_authority.cache_aliases.empty()) {
        maintenance = amdgpu::DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
            effect_domains, effect_authority.cache_aliases, effect_authority.resolve_writeback,
            effect_authority.resolve_witness);
      } else if (resolve_cache_writeback_) {
        maintenance = amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
            effect_domains, amdgpu::DeviceCacheOperation::WritebackInvalidate,
            resolve_cache_writeback_);
      } else {
        maintenance = amdgpu::DeviceCacheCoherence::acquire_cache_maintenance_group(
            coherence_domains_, amdgpu::DeviceCacheOperation::WritebackInvalidate);
      }
    } catch (const amdgpu::CacheMaintenanceUnavailable &) {
      release_maintenance();
      effect_authority.permit.reset();
      if (retry_request(request))
        return;
      Request::EffectPhase expected = Request::EffectPhase::Entered;
      request->phase.compare_exchange_strong(expected, Request::EffectPhase::Revoked,
                                             std::memory_order_acq_rel);
      finish(request->id, VmAccessOutcome::Revoked);
      return;
    } catch (...) {
      // A missing maintenance backing or failed dirty writeback is terminal for
      // this request. The VM gate has not entered and no fabric backing effect ran.
      release_maintenance();
      maintenance_ready = false;
    }
  }

  if (!maintenance_ready) {
    request->effect_outcome = authority_ready ? VmAccessOutcome::Faulted
                              : effect_authority.outcome == VmAccessOutcome::Complete
                                  ? VmAccessOutcome::Revoked
                                  : effect_authority.outcome;
    Request::EffectPhase expected = Request::EffectPhase::Entered;
    request->phase.compare_exchange_strong(expected, Request::EffectPhase::Done,
                                           std::memory_order_acq_rel);
  } else {
    CanonicalLineFreshness::WriterAuthority writer = line_freshness_.try_acquire_writer();
    if (!writer) {
      release_maintenance();
      effect_authority.permit.reset();
      if (retry_request(request))
        return;
      Request::EffectPhase expected = Request::EffectPhase::Entered;
      request->phase.compare_exchange_strong(expected, Request::EffectPhase::Revoked,
                                             std::memory_order_acq_rel);
      finish(request->id, VmAccessOutcome::Revoked);
      return;
    }
    if (!coherence_domains_.empty())
      writer.invalidate_all();
    bool gate_entered = false;
    try {
      gate_entered = request->effect_gate->commit([&] {
        if (request->phase.load(std::memory_order_acquire) != Request::EffectPhase::Entered)
          return;
        if (effect_probe_for_testing_)
          effect_probe_for_testing_();
        Request::EffectPhase expected = Request::EffectPhase::Entered;
        if (!request->phase.compare_exchange_strong(expected, Request::EffectPhase::Committing,
                                                    std::memory_order_acq_rel))
          return;
        const bool complete = request->write ? request->backing.write(request->payload)
                                             : request->backing.read(request->effect_response);
        if (complete) {
          if (request->write) {
            writer.advance(request->backing.backing_id(), request->backing.offset(),
                           request->backing.extent());
          } else {
            request->read_stamp = line_freshness_.capture_read_under_writer(
                writer, request->backing.backing_id(), request->backing.offset(),
                request->backing.extent());
          }
        }
        writer = {};
        // Releasing inside the gate publishes all SoC epochs before the
        // committed outcome becomes visible to polling or reconciliation.
        release_maintenance();
        request->effect_outcome = complete ? VmAccessOutcome::Complete : VmAccessOutcome::Faulted;
        if (complete) {
          std::lock_guard state_lock(state_->mutex);
          ++state_->diagnostics.effects;
          if (!request->local) {
            const std::size_t direction = request->step < 0 ? 1 : 0;
            auto &directed =
                state_->diagnostics.directed_effects[request->route_index * 2 + direction];
            ++(request->write ? directed.writes : directed.reads);
            (request->write ? directed.write_bytes : directed.read_bytes) +=
                request->backing.extent();
            if (state_->effect_trace_capacity != 0) {
              auto &trace = state_->diagnostics.committed_effects;
              if (trace.size() < state_->effect_trace_capacity) {
                trace.push_back({.request_id = request->id,
                                 .backing_id = request->backing.backing_id(),
                                 .backing_offset = request->backing.offset(),
                                 .bytes = request->backing.extent(),
                                 .requester_gpu_id = directed.requester_gpu_id,
                                 .owner_gpu_id = directed.owner_gpu_id,
                                 .write = request->write,
                                 .origin = request->trace_origin});
              } else {
                ++state_->diagnostics.effect_trace_dropped;
              }
            }
          }
        }
        request->phase.store(Request::EffectPhase::Done, std::memory_order_release);
      });
    } catch (...) {
      release_maintenance();
      const Request::EffectPhase phase = request->phase.load(std::memory_order_acquire);
      if (phase == Request::EffectPhase::Entered) {
        request->effect_outcome = VmAccessOutcome::Faulted;
        Request::EffectPhase expected = Request::EffectPhase::Entered;
        request->phase.compare_exchange_strong(expected, Request::EffectPhase::Done,
                                               std::memory_order_acq_rel);
      } else if (phase == Request::EffectPhase::Committing) {
        // Backing already returned and published its result. Retain that result
        // if later bookkeeping failed, so reconciliation cannot replay it.
        request->phase.store(Request::EffectPhase::Done, std::memory_order_release);
      }
    }
    release_maintenance();
    if (!gate_entered) {
      // The VM gate closed before entry. No backing effect was possible.
      Request::EffectPhase expected = Request::EffectPhase::Entered;
      request->phase.compare_exchange_strong(expected, Request::EffectPhase::Revoked,
                                             std::memory_order_acq_rel);
    }
  }
  effect_authority.permit.reset();
  // The reservation spans admission retries but no terminal reply. The target
  // handler still owns the active guard while the release callback runs.
  request->request_reservation.reset();
  VmAccessOutcome effect_outcome;
  std::vector<std::byte> response;
  const Request::EffectPhase phase = request->phase.load(std::memory_order_acquire);
  effect_outcome =
      phase == Request::EffectPhase::Done ? request->effect_outcome : VmAccessOutcome::Revoked;
  if (phase == Request::EffectPhase::Done)
    response = request->effect_response;
  if (request->local) {
    finish(request->id, effect_outcome);
  } else {
    WireMessage reply(message->id, message->route, message->position, -message->step, true,
                      request->write, effect_outcome, std::move(response));
    send_next(wired_routes_[message->route], reply);
  }
}

void FabricEndpoint::finish(uint64_t request_id, VmAccessOutcome outcome) {
  std::shared_ptr<Request> request;
  amdgpu::MemoryIngressPermit reservation;
  {
    std::lock_guard lock(state_->mutex);
    const auto found = state_->pending.find(request_id);
    if (found == state_->pending.end())
      return;
    request = std::move(found->second);
    state_->pending.erase(found);
    reservation = state_->take_idle_reservation(*request);
  }
  state_->finish_reservation_release(reservation);
  std::lock_guard lock(request->mutex);
  if (request->outcome == VmAccessOutcome::Unavailable) {
    const Request::EffectPhase phase = request->phase.load(std::memory_order_acquire);
    request->outcome =
        phase == Request::EffectPhase::Done ? request->effect_outcome
        : phase == Request::EffectPhase::Entered || phase == Request::EffectPhase::Committing
            ? VmAccessOutcome::Unavailable
            : outcome;
    if (phase == Request::EffectPhase::Done && !request->write &&
        request->effect_outcome == VmAccessOutcome::Complete)
      request->response = request->effect_response;
  }
}

FabricEndpoint::Diagnostics FabricEndpoint::diagnostics() const {
  Diagnostics snapshot;
  {
    std::lock_guard lock(state_->mutex);
    snapshot = state_->diagnostics;
  }
  snapshot.scoped_maintenance.reserve(coherence_domains_.size());
  for (size_t index = 0; index < coherence_domains_.size(); ++index) {
    const auto &domain = coherence_domains_[index];
    snapshot.scoped_maintenance.push_back({
        .domain_index = index,
        .gpu_id = prepared_domains_.empty() ? 0 : prepared_domains_[index].gpu_id,
        .successes = domain->scoped_maintenance_success_count(),
        .admission_timeouts = domain->scoped_admission_timeout_count(),
    });
  }
  return snapshot;
}

bool FabricEndpoint::enable_effect_trace(std::size_t capacity) {
  std::lock_guard lock(state_->mutex);
  if (capacity == 0 || state_->effect_trace_capacity != 0 || state_->diagnostics.submissions != 0 ||
      !state_->pending.empty())
    return false;
  state_->diagnostics.committed_effects.reserve(capacity);
  state_->effect_trace_capacity = capacity;
  return true;
}

} // namespace rocjitsu
