// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file system_graph.h
/// @brief Owned, validated description of system nodes and fabric connectivity.

#pragma once

#include "rocjitsu/config/kfd_device_config.h"
#include "util/diagnostic.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::fb {
struct SimulationConfig;
} // namespace rocjitsu::fb

namespace rocjitsu::config {

/// A global instance name and its node-local KFD identity.
struct SystemGpuInstance {
  std::string id;
  std::string profile; ///< "vm" uses the top-level vm and topology template.
  KfdDeviceConfig device;
};

struct SystemNode {
  std::string id;
  std::vector<SystemGpuInstance> gpus; ///< Stable local ordinal order.
};

enum class SystemEndpointKind { Gpu, Switch, Network };

struct SystemEndpoint {
  std::string id;
  SystemEndpointKind kind = SystemEndpointKind::Gpu;
  std::string gpu;  ///< Set only for GPU endpoints.
  std::string node; ///< Set only for network endpoints.
};

enum class SystemLinkClass { ScaleUp, Net, InfinityFabric };

/// A bidirectional descriptive link. It grants no peer-memory access.
struct SystemLink {
  std::string src;
  std::string dst;
  SystemLinkClass link_class = SystemLinkClass::ScaleUp;
};

struct SystemCoherenceDomain {
  std::string id;
  std::vector<std::string> gpus;
};

/// Bounds the synchronous fabric handler chain built from a projected route.
inline constexpr std::size_t kMaxPeerRouteEndpoints = 64;

/// One unambiguous local fabric path. Only the first and last endpoints are
/// GPUs; intermediate endpoint IDs identify switches. All values are owned.
struct SystemPeerRoute {
  std::string source_gpu;
  std::string target_gpu;
  uint32_t source_gpu_id = 0;
  uint32_t target_gpu_id = 0;
  std::vector<std::string> endpoints;
  /// Local fabric class for each consecutive pair of endpoints, in path order.
  std::vector<SystemLinkClass> hop_classes{};
};

/// An owned, read-only view of qualified routes for one selected node.
class SystemPeerRouteProjection {
public:
  SystemPeerRouteProjection() = default;

  std::string_view node_id() const { return node_id_; }
  std::span<const SystemPeerRoute> routes() const { return routes_; }

private:
  friend class SystemGraph;
  SystemPeerRouteProjection(std::string node_id, std::vector<SystemPeerRoute> routes)
      : node_id_(std::move(node_id)), routes_(std::move(routes)) {}

  std::string node_id_;
  std::vector<SystemPeerRoute> routes_;
};

/// The graph owns every value copied from the FlatBuffer parser.
class SystemGraph {
public:
  bool explicit_manifest = false;
  std::vector<SystemNode> nodes;
  std::vector<SystemEndpoint> endpoints;
  std::vector<SystemLink> links;
  std::vector<SystemCoherenceDomain> coherence_domains;

  /// Select one local node. An empty selector is valid only for one-node input.
  /// Returns failure with one diagnostic for an unknown or ambiguous selection.
  /// The returned pointer borrows this graph. Allocation failures may still throw.
  util::FailureOr<const SystemNode *> select_node(util::DiagnosticEmitter emit_error,
                                                  std::string_view node_id = {}) const;

  /// Return local GPU instances in stable node order, borrowing this graph.
  /// Returns failure with one diagnostic when node selection fails.
  util::FailureOr<std::span<const SystemGpuInstance>>
  local_gpus(util::DiagnosticEmitter emit_error, std::string_view node_id = {}) const;

  /// Returns failure with one diagnostic when node selection fails.
  /// Copy eligible local peer paths. Descriptive legacy links never qualify.
  /// Missing, ambiguous, GPU-transit, remote-tainted, or overlong paths grant
  /// no route.
  util::FailureOr<SystemPeerRouteProjection> local_peer_routes(util::DiagnosticEmitter emit_error,
                                                               std::string_view node_id = {}) const;
};

/// Normalize and validate an in-scope FlatBuffers configuration. The returned
/// graph does not retain any FlatBuffers pointers. Invalid input returns failure
/// after emitting one diagnostic. Allocation and library failures may still throw.
util::FailureOr<SystemGraph> system_graph_from_fb(const fb::SimulationConfig *config,
                                                  util::DiagnosticEmitter emit_error);

/// Parse a JSON file or string through the supplied FlatBuffers schema.
util::FailureOr<SystemGraph> load_system_graph(const std::string &json_path,
                                               const std::string &schema_text,
                                               util::DiagnosticEmitter emit_error);
util::FailureOr<SystemGraph> load_system_graph_from_string(const std::string &json,
                                                           const std::string &schema_text,
                                                           util::DiagnosticEmitter emit_error);

} // namespace rocjitsu::config
