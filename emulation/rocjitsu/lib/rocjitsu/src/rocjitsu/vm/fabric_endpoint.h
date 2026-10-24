// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file fabric_endpoint.h
/// @brief Owned local ScaleUp and Infinity Fabric requests and their Simdojo graph.

#pragma once

#include "rocjitsu/config/system_graph.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/canonical_line_freshness.h"
#include "rocjitsu/vm/fabric_backing.h"

#include "simdojo/sim/component.h"
#include "simdojo/sim/exec_mode.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace simdojo {
class Topology;
} // namespace simdojo

namespace rocjitsu {

/// Owner-local effects and local switched peer transport. Constructing this
/// graph alone never grants a mapping; KFD and each caller control admission.
/// With SoC coherence domains, every admitted VFIO peer effect conservatively
/// writes back and invalidates the participating caches and invalidates the
/// freshness table before backing access. A peer read can therefore observe
/// owner dirty data without guest HDP maintenance. This is a functional
/// coherence guarantee for UC PCIe peer mappings, not a validation of guest
/// cache-maintenance ordering or native xGMI/cacheable peer behavior.
class FabricEndpoint final : public simdojo::CompositeComponent {
public:
  /// @brief Authority retained from owner-thread admission through the effect.
  /// @details A KFD provider may select every live alias of the backing and
  /// retain its publication and ingress exclusion in @p permit. A manual
  /// endpoint may return only a requester permit and use full-domain cache
  /// maintenance. The selected resolver can enter the provider's exact
  /// effect window while dirty cache lines are prepared.
  class EffectAuthority {
  public:
    EffectAuthority() = default;
    explicit EffectAuthority(amdgpu::MemoryIngressAdmission admission)
        : outcome(admission.outcome), permit(std::move(admission.permit)) {}

    amdgpu::VmAccessOutcome outcome = amdgpu::VmAccessOutcome::Complete;
    amdgpu::MemoryIngressPermit permit;
    std::vector<amdgpu::DeviceCacheAliasScope> cache_aliases;
    amdgpu::PreparedWritebackResolver resolve_writeback;
    amdgpu::CanonicalLineWitnessResolver resolve_witness;
  };

  /// The request reservation survives a temporarily unavailable admission and
  /// is released after the terminal target handler has finished its effect.
  using EffectAdmission = std::function<EffectAuthority(
      const FabricBacking &, bool, amdgpu::MemoryIngressPermit &request_reservation)>;

  /// @brief Admit one target-handler attempt before it enters the owner effect.
  /// @details A successful permit spans validation, backing access, and handler
  /// cleanup. A temporary refusal retries without reading target-side policy.
  /// Unlike the request reservation, this permit never survives a retry.
  using PreEffectAdmission = std::function<amdgpu::MemoryIngressAdmission()>;

  class PreparedTransfer {
  public:
    amdgpu::VmAccessOutcome outcome = amdgpu::VmAccessOutcome::Faulted;
    std::unique_ptr<amdgpu::VmDeferredTransfer> operation;
    explicit operator bool() const { return operation != nullptr; }
  };

  /// Optional requester coordinates for diagnostics only. The endpoint never
  /// uses these fields for route, backing, or effect admission.
  struct TraceOrigin {
    uint64_t map_id;
    uint64_t gpu_va;
    uint64_t iova;
  };

  struct Diagnostics {
    /// Successful backing accesses on each route. Byte totals exclude failed
    /// or revoked effects and do not identify application payloads.
    struct DirectedEffects {
      uint32_t requester_gpu_id = 0;
      uint32_t owner_gpu_id = 0;
      uint64_t reads = 0;
      uint64_t read_bytes = 0;
      uint64_t writes = 0;
      uint64_t write_bytes = 0;
    };

    /// One successful directed backing access, in owner-effect commit order.
    /// A backing ID of zero is anonymous. These coordinates do not identify
    /// an application payload or the guest mapping that admitted the effect.
    struct CommittedEffect {
      uint64_t request_id = 0;
      uint64_t backing_id = 0;
      uint64_t backing_offset = 0;
      uint64_t bytes = 0;
      uint32_t requester_gpu_id = 0;
      uint32_t owner_gpu_id = 0;
      bool write = false;
      TraceOrigin origin{};
    };

    uint64_t submissions = 0;
    uint64_t request_messages = 0;
    uint64_t response_messages = 0;
    uint64_t effects = 0;
    std::vector<DirectedEffects> directed_effects;
    std::vector<CommittedEffect> committed_effects;
    uint64_t effect_trace_dropped = 0;
    std::thread::id owner_thread;
    std::thread::id handler_thread;
  };

  /// @brief Build local peer routes with optional SoC cache-coherence domains.
  /// @details Distinct, non-null domains are supplied in stable local SoC
  /// order. Each admitted backing effect quiesces all supplied domains on the
  /// fabric owner thread. An empty list retains the manual endpoint's
  /// direct-backing behavior.
  explicit FabricEndpoint(
      std::vector<config::SystemPeerRoute> routes,
      std::vector<std::shared_ptr<amdgpu::DeviceCacheCoherence>> coherence_domains = {});
  ~FabricEndpoint() override;

  FabricEndpoint(const FabricEndpoint &) = delete;
  FabricEndpoint &operator=(const FabricEndpoint &) = delete;

  /// Add a separate incoming and outgoing port for every physical edge.
  /// Call once after this component has been attached to the topology root.
  void wire(simdojo::Topology &topology, simdojo::ExecMode mode);

  /// The returned operation owns its request and can be wrapped in a
  /// VmTransferToken by a later translator. A missing/forbidden route returns
  /// Faulted, never an empty optional that permits direct backing fallback.
  [[nodiscard]] PreparedTransfer
  prepare_read(uint32_t requester_gpu_id, uint32_t owner_gpu_id, FabricBacking backing,
               std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate,
               EffectAdmission admit_effect = {}, PreEffectAdmission admit_pre_effect = {},
               TraceOrigin trace_origin = {});
  [[nodiscard]] PreparedTransfer
  prepare_write(uint32_t requester_gpu_id, uint32_t owner_gpu_id, FabricBacking backing,
                std::span<const std::byte> bytes,
                std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate,
                EffectAdmission admit_effect = {}, PreEffectAdmission admit_pre_effect = {},
                TraceOrigin trace_origin = {});
  /// Submit an owner-local write through the same admission, cache maintenance,
  /// VM gate, and reconciliation path as a peer effect. A non-empty admission
  /// callback is required. No route or link is needed; the owner inbox invokes
  /// the target handler directly.
  [[nodiscard]] PreparedTransfer prepare_local_write(
      uint32_t owner_gpu_id, FabricBacking backing, std::span<const std::byte> bytes,
      std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate, EffectAdmission admit_effect);

  /// @brief True only for one validated local route between these GPUs.
  /// @details This is a bounded policy query and does not acquire backing or
  /// submit a request. Call after wire(), while the endpoint is alive.
  [[nodiscard]] bool can_route(uint32_t requester_gpu_id, uint32_t owner_gpu_id) const;

  [[nodiscard]] std::span<const config::SystemPeerRoute> routes() const { return routes_; }
  [[nodiscard]] Diagnostics diagnostics() const;

  /// Reserve bounded, allocation-free commit records before any request is
  /// submitted. Return false if tracing was already configured or work began.
  /// A full trace increments effect_trace_dropped instead of losing that fact.
  [[nodiscard]] bool enable_effect_trace(std::size_t capacity);

  /// Test seam for pausing an effect after it entered the endpoint's drain
  /// permit. Set only before submitting requests; the callback must not
  /// re-enter this endpoint.
  void set_effect_probe_for_testing(std::function<void()> probe) {
    effect_probe_for_testing_ = std::move(probe);
  }

  void initialize() override;
  void shutdown() override;

private:
  friend class VirtualMachine;

  class Request;
  class State;
  struct Route;
  class WireMessage;
  class Node;
  class Operation;

  [[nodiscard]] PreparedTransfer
  prepare(uint32_t requester_gpu_id, uint32_t owner_gpu_id, FabricBacking backing,
          std::span<const std::byte> bytes, bool write, bool local,
          std::shared_ptr<amdgpu::VmTransferEffectGate> effect_gate, EffectAdmission admit_effect,
          PreEffectAdmission admit_pre_effect, TraceOrigin trace_origin);
  void service_inbox();
  void receive(Node &node, WireMessage &message);
  void handle_target(const std::shared_ptr<Request> &request, const WireMessage *message);
  [[nodiscard]] bool retry_request(const std::shared_ptr<Request> &request) noexcept;
  void finish(uint64_t request_id, amdgpu::VmAccessOutcome outcome);
  void send_next(const Route &route, WireMessage &message);
  /// Configure the KFD-backed production provider before wiring the endpoint.
  void configure_cache_writeback(std::vector<amdgpu::DeviceCacheDomain> domains,
                                 amdgpu::PreparedWritebackResolver resolver);

  std::vector<config::SystemPeerRoute> routes_;
  std::vector<Route> wired_routes_;
  std::vector<std::shared_ptr<amdgpu::DeviceCacheCoherence>> coherence_domains_;
  std::vector<amdgpu::DeviceCacheDomain> prepared_domains_;
  amdgpu::PreparedWritebackResolver resolve_cache_writeback_;
  CanonicalLineFreshness line_freshness_;
  std::shared_ptr<State> state_;
  simdojo::Event inbox_event_{this, simdojo::EventType::TIMER_CALLBACK};
  std::function<void()> effect_probe_for_testing_;
  bool wired_ = false;
};

} // namespace rocjitsu
