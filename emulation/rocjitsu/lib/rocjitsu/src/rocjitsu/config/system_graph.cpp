// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/config/system_graph.h"

#include "rocjitsu/config/config_common.h"

#include "simulation_config_generated.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu::config {
namespace {

util::FailureOr<std::string> required_name(const flatbuffers::String *name, std::string_view field,
                                           util::DiagnosticEmitter emit_error) {
  if (!name || name->str().empty())
    return emit_error.emit() << field << " must be nonempty";
  return name->str();
}

using LocalFabricAdjacency = std::vector<std::vector<size_t>>;
constexpr size_t kNoEndpoint = std::numeric_limits<size_t>::max();

std::pair<size_t, size_t> edge_key(size_t a, size_t b) { return std::minmax(a, b); }

// Only switches may be interior vertices. An omitted edge lets the caller
// detect a second simple path without enumerating every path through cycles.
std::vector<size_t> find_peer_path(const SystemGraph &graph, const LocalFabricAdjacency &adjacency,
                                   const std::vector<bool> &unsafe_switches, size_t source,
                                   size_t target,
                                   std::pair<size_t, size_t> omitted = {kNoEndpoint, kNoEndpoint}) {
  std::vector<size_t> parent(graph.endpoints.size(), kNoEndpoint);
  std::vector<size_t> queue{source};
  parent[source] = source;
  for (size_t head = 0; head < queue.size() && parent[target] == kNoEndpoint; ++head) {
    const size_t current = queue[head];
    for (size_t neighbor : adjacency[current]) {
      if (parent[neighbor] != kNoEndpoint || edge_key(current, neighbor) == omitted)
        continue;
      if (neighbor != target && (graph.endpoints[neighbor].kind != SystemEndpointKind::Switch ||
                                 unsafe_switches[neighbor]))
        continue;
      parent[neighbor] = current;
      queue.push_back(neighbor);
    }
  }
  if (parent[target] == kNoEndpoint)
    return {};
  std::vector<size_t> path;
  for (size_t current = target; current != source; current = parent[current])
    path.push_back(current);
  path.push_back(source);
  std::reverse(path.begin(), path.end());
  return path;
}

bool has_alternate_peer_path(const SystemGraph &graph, const LocalFabricAdjacency &adjacency,
                             const std::vector<bool> &unsafe_switches, size_t source, size_t target,
                             const std::vector<size_t> &first_path) {
  for (size_t i = 1; i < first_path.size(); ++i)
    if (!find_peer_path(graph, adjacency, unsafe_switches, source, target,
                        edge_key(first_path[i - 1], first_path[i]))
             .empty())
      return true;
  return false;
}

util::FailureOr<SystemGraph> legacy_graph(const fb::SimulationConfig *config,
                                          util::DiagnosticEmitter emit_error) {
  SystemGraph graph;
  SystemNode node;
  node.id = "node0";
  const auto *gpu = config->vm() ? config->vm()->gpu() : nullptr;
  const bool has_device = gpu && gpu->device();
  const uint32_t count = has_device ? std::max(1u, gpu->num_gpus()) : 1u;
  auto base = kfd_device_from_fb(has_device ? gpu->device() : nullptr, "vm.gpu.device", emit_error);
  if (base.failed())
    return util::Result::failure();

  for (uint32_t i = 0; i < count; ++i) {
    SystemGpuInstance instance;
    instance.id = "gpu" + std::to_string(i);
    instance.profile = "vm";
    instance.device = base.value();
    if (count > 1) {
      // Match the existing loader's identity expansion exactly.
      instance.device.gpu_id = base.value().gpu_id + i;
      instance.device.location_id = 0x0300 + (i << 8);
      instance.device.drm_render_minor = 128 + i;
      instance.device.unique_id = base.value().unique_id + i;
    }
    graph.endpoints.push_back({instance.id, SystemEndpointKind::Gpu, instance.id, {}});
    node.gpus.push_back(std::move(instance));
  }
  for (size_t i = 0; i < node.gpus.size(); ++i)
    for (size_t j = i + 1; j < node.gpus.size(); ++j)
      graph.links.push_back({node.gpus[i].id, node.gpus[j].id, SystemLinkClass::ScaleUp});
  graph.nodes.push_back(std::move(node));
  return graph;
}

util::FailureOr<SystemGraph> manifest_graph(const fb::SimulationConfig *config,
                                            util::DiagnosticEmitter emit_error) {
  const auto *manifest = config->system();
  if (!config->vm() || !config->vm()->gpu() || !config->vm()->gpu()->device() ||
      !config->topology())
    return emit_error.emit() << "system requires a top-level vm.gpu.device and topology profile";
  if (config->vm()->gpu()->num_gpus() != 1)
    return emit_error.emit() << "system cannot be combined with vm.gpu.num_gpus replication";
  if (config->dbt_guest() && config->dbt_guest()->enabled())
    return emit_error.emit() << "system cannot be combined with dbt_guest mode";
  if (!manifest->nodes() || manifest->nodes()->size() == 0)
    return emit_error.emit() << "system.nodes must contain at least one node";

  SystemGraph graph;
  graph.explicit_manifest = true;
  auto base = kfd_device_from_fb(config->vm()->gpu()->device(), "vm.gpu.device", emit_error);
  if (base.failed())
    return util::Result::failure();
  std::unordered_set<std::string> node_ids;
  std::unordered_map<std::string, std::string> gpu_nodes;
  std::unordered_set<uint64_t> unique_ids;
  for (const auto *node_def : *manifest->nodes()) {
    SystemNode node;
    auto name = required_name(node_def->id(), "system.nodes[].id", emit_error);
    if (name.failed())
      return util::Result::failure();
    node.id = std::move(name.value());
    if (!node_ids.insert(node.id).second)
      return emit_error.emit() << "duplicate system node id: " << node.id;
    if (!node_def->gpus() || node_def->gpus()->size() == 0)
      return emit_error.emit() << "system node has no GPUs: " << node.id;

    std::unordered_set<uint32_t> gpu_ids;
    std::unordered_set<uint32_t> locations;
    std::unordered_set<uint32_t> render_minors;
    for (const auto *gpu_def : *node_def->gpus()) {
      SystemGpuInstance instance;
      auto name = required_name(gpu_def->id(), "system.nodes[].gpus[].id", emit_error);
      if (name.failed())
        return util::Result::failure();
      instance.id = std::move(name.value());
      if (!gpu_nodes.emplace(instance.id, node.id).second)
        return emit_error.emit() << "duplicate system GPU instance id: " << instance.id;
      instance.profile = gpu_def->profile() ? gpu_def->profile()->str() : "vm";
      if (instance.profile != "vm")
        return emit_error.emit() << "unsupported GPU profile for " << instance.id << ": "
                                 << instance.profile;
      instance.device = base.value();
      instance.device.gpu_id = gpu_def->gpu_id();
      instance.device.unique_id = gpu_def->unique_id();
      instance.device.location_id = gpu_def->location_id();
      instance.device.drm_render_minor = gpu_def->drm_render_minor();
      if (!instance.device.gpu_id || !gpu_ids.insert(instance.device.gpu_id).second)
        return emit_error.emit() << "system node has missing or duplicate KFD GPU ID: " << node.id;
      if (!instance.device.unique_id)
        return emit_error.emit() << "system GPU instance has no unique ID: " << instance.id;
      if (!unique_ids.insert(instance.device.unique_id).second)
        return emit_error.emit() << "duplicate system GPU unique ID: " << instance.device.unique_id;
      if (!instance.device.location_id || !locations.insert(instance.device.location_id).second)
        return emit_error.emit() << "system node has missing or duplicate PCI location: "
                                 << node.id;
      if (instance.device.drm_render_minor < 128 ||
          !render_minors.insert(instance.device.drm_render_minor).second)
        return emit_error.emit() << "system node has invalid or duplicate DRM render minor: "
                                 << node.id;
      node.gpus.push_back(std::move(instance));
    }
    graph.nodes.push_back(std::move(node));
  }

  std::unordered_map<std::string, size_t> endpoint_indices;
  if (manifest->endpoints())
    for (const auto *endpoint_def : *manifest->endpoints()) {
      SystemEndpoint endpoint;
      auto name = required_name(endpoint_def->id(), "system.endpoints[].id", emit_error);
      if (name.failed())
        return util::Result::failure();
      endpoint.id = std::move(name.value());
      if (!endpoint_indices.emplace(endpoint.id, graph.endpoints.size()).second)
        return emit_error.emit() << "duplicate system endpoint id: " << endpoint.id;
      switch (endpoint_def->kind()) {
      case fb::SystemEndpointKind_gpu: {
        endpoint.kind = SystemEndpointKind::Gpu;
        auto name = required_name(endpoint_def->gpu(), "GPU endpoint gpu", emit_error);
        if (name.failed())
          return util::Result::failure();
        endpoint.gpu = std::move(name.value());
        if (!gpu_nodes.contains(endpoint.gpu) || endpoint_def->node())
          return emit_error.emit() << "invalid GPU endpoint: " << endpoint.id;
        break;
      }
      case fb::SystemEndpointKind_switch_:
        endpoint.kind = SystemEndpointKind::Switch;
        if (endpoint_def->gpu() || endpoint_def->node())
          return emit_error.emit() << "switch endpoint cannot name a GPU or node: " << endpoint.id;
        break;
      case fb::SystemEndpointKind_network: {
        endpoint.kind = SystemEndpointKind::Network;
        auto name = required_name(endpoint_def->node(), "network endpoint node", emit_error);
        if (name.failed())
          return util::Result::failure();
        endpoint.node = std::move(name.value());
        if (!node_ids.contains(endpoint.node) || endpoint_def->gpu())
          return emit_error.emit() << "invalid network endpoint: " << endpoint.id;
        break;
      }
      default:
        return emit_error.emit() << "invalid system endpoint kind: " << endpoint.id;
      }
      graph.endpoints.push_back(std::move(endpoint));
    }

  std::set<std::pair<std::string, std::string>> links_seen;
  if (manifest->links())
    for (const auto *link_def : *manifest->links()) {
      SystemLink link;
      auto src_name = required_name(link_def->src(), "system.links[].src", emit_error);
      if (src_name.failed())
        return util::Result::failure();
      link.src = std::move(src_name.value());
      auto dst_name = required_name(link_def->dst(), "system.links[].dst", emit_error);
      if (dst_name.failed())
        return util::Result::failure();
      link.dst = std::move(dst_name.value());
      if (link.src == link.dst || !endpoint_indices.contains(link.src) ||
          !endpoint_indices.contains(link.dst))
        return emit_error.emit() << "invalid system link: " << link.src << " -> " << link.dst;
      // Both endpoint memberships were checked above.
      const auto &src = graph.endpoints[endpoint_indices.at(link.src)];
      const auto &dst = graph.endpoints[endpoint_indices.at(link.dst)];
      switch (link_def->link_class()) {
      case fb::SystemLinkClass_scale_up:
        link.link_class = SystemLinkClass::ScaleUp;
        if (src.kind == SystemEndpointKind::Network || dst.kind == SystemEndpointKind::Network)
          return emit_error.emit() << "scale-up link has a network endpoint";
        break;
      case fb::SystemLinkClass_infinity_fabric:
        link.link_class = SystemLinkClass::InfinityFabric;
        if (src.kind == SystemEndpointKind::Network || dst.kind == SystemEndpointKind::Network)
          return emit_error.emit() << "Infinity Fabric link has a network endpoint";
        break;
      case fb::SystemLinkClass_net:
        link.link_class = SystemLinkClass::Net;
        if (src.kind != SystemEndpointKind::Network || dst.kind != SystemEndpointKind::Network ||
            src.node == dst.node)
          return emit_error.emit() << "NET link must join different system nodes";
        break;
      default:
        return emit_error.emit() << "invalid system link class";
      }
      const auto &[first, second] = std::minmax(link.src, link.dst);
      if (!links_seen.emplace(first, second).second)
        return emit_error.emit() << "duplicate system link: " << first << " <-> " << second;
      graph.links.push_back(std::move(link));
    }

  std::unordered_set<std::string> domain_ids;
  std::unordered_set<std::string> assigned_gpus;
  if (manifest->coherence_domains())
    for (const auto *domain_def : *manifest->coherence_domains()) {
      SystemCoherenceDomain domain;
      auto name = required_name(domain_def->id(), "system.coherence_domains[].id", emit_error);
      if (name.failed())
        return util::Result::failure();
      domain.id = std::move(name.value());
      if (!domain_ids.insert(domain.id).second)
        return emit_error.emit() << "duplicate coherence domain id: " << domain.id;
      if (!domain_def->gpus() || domain_def->gpus()->size() == 0)
        return emit_error.emit() << "coherence domain has no GPUs: " << domain.id;
      std::string owner_node;
      for (const auto *member : *domain_def->gpus()) {
        auto name = required_name(member, "coherence domain GPU", emit_error);
        if (name.failed())
          return util::Result::failure();
        const std::string gpu = std::move(name.value());
        const auto found = gpu_nodes.find(gpu);
        if (found == gpu_nodes.end() || !assigned_gpus.insert(gpu).second)
          return emit_error.emit() << "unknown or repeated coherence-domain GPU: " << gpu;
        if (!owner_node.empty() && owner_node != found->second)
          return emit_error.emit() << "coherence domain spans system nodes: " << domain.id;
        owner_node = found->second;
        domain.gpus.push_back(gpu);
      }
      graph.coherence_domains.push_back(std::move(domain));
    }
  return graph;
}

} // namespace

util::FailureOr<const SystemNode *> SystemGraph::select_node(util::DiagnosticEmitter emit_error,
                                                             std::string_view node_id) const {
  if (node_id.empty()) {
    if (nodes.size() != 1)
      return emit_error.emit() << "multi-node system requires a local node ID";
    return &nodes.front();
  }
  const auto it = std::find_if(nodes.begin(), nodes.end(),
                               [&](const SystemNode &node) { return node.id == node_id; });
  if (it == nodes.end())
    return emit_error.emit() << "unknown local node ID: " << node_id;
  return &*it;
}

util::FailureOr<std::span<const SystemGpuInstance>>
SystemGraph::local_gpus(util::DiagnosticEmitter emit_error, std::string_view node_id) const {
  auto node = select_node(emit_error, node_id);
  if (node.failed())
    return util::Result::failure();
  return std::span<const SystemGpuInstance>(node.value()->gpus);
}

util::FailureOr<SystemPeerRouteProjection>
SystemGraph::local_peer_routes(util::DiagnosticEmitter emit_error, std::string_view node_id) const {
  auto node = select_node(emit_error, node_id);
  if (node.failed())
    return util::Result::failure();
  const SystemNode &selected = *node.value();
  if (!explicit_manifest)
    return SystemPeerRouteProjection(selected.id, {});

  std::unordered_set<std::string> local_gpu_names;
  for (const auto &gpu : selected.gpus)
    local_gpu_names.insert(gpu.id);

  std::unordered_map<std::string, size_t> endpoint_indices;
  std::unordered_map<std::string, std::vector<size_t>> gpu_endpoints;
  for (size_t i = 0; i < endpoints.size(); ++i) {
    endpoint_indices.emplace(endpoints[i].id, i);
    if (endpoints[i].kind == SystemEndpointKind::Gpu && local_gpu_names.contains(endpoints[i].gpu))
      gpu_endpoints[endpoints[i].gpu].push_back(i);
  }

  LocalFabricAdjacency adjacency(endpoints.size());
  for (const auto &link : links) {
    if (link.link_class != SystemLinkClass::ScaleUp &&
        link.link_class != SystemLinkClass::InfinityFabric)
      continue;
    const auto source = endpoint_indices.find(link.src);
    const auto target = endpoint_indices.find(link.dst);
    if (source == endpoint_indices.end() || target == endpoint_indices.end())
      continue;
    adjacency[source->second].push_back(target->second);
    adjacency[target->second].push_back(source->second);
  }

  // Switches are not node-labeled. Exclude the entire switch component when
  // any of its local fabric edges reaches a GPU outside the selected node.
  std::vector<bool> unsafe_switches(endpoints.size(), false);
  std::vector<bool> seen_switches(endpoints.size(), false);
  for (size_t i = 0; i < endpoints.size(); ++i) {
    if (endpoints[i].kind != SystemEndpointKind::Switch || seen_switches[i])
      continue;
    std::vector<size_t> component{i};
    seen_switches[i] = true;
    bool unsafe = false;
    for (size_t head = 0; head < component.size(); ++head) {
      for (size_t neighbor : adjacency[component[head]]) {
        const auto &endpoint = endpoints[neighbor];
        if (endpoint.kind == SystemEndpointKind::Switch) {
          if (!seen_switches[neighbor]) {
            seen_switches[neighbor] = true;
            component.push_back(neighbor);
          }
        } else if (endpoint.kind != SystemEndpointKind::Gpu ||
                   !local_gpu_names.contains(endpoint.gpu)) {
          unsafe = true;
        }
      }
    }
    for (size_t member : component)
      unsafe_switches[member] = unsafe;
  }

  std::vector<SystemPeerRoute> routes;
  for (size_t source = 0; source < selected.gpus.size(); ++source) {
    const auto &source_gpu = selected.gpus[source];
    for (size_t target = source + 1; target < selected.gpus.size(); ++target) {
      const auto &target_gpu = selected.gpus[target];
      std::vector<size_t> unique_path;
      bool ambiguous = false;
      for (size_t source_endpoint : gpu_endpoints[source_gpu.id]) {
        for (size_t target_endpoint : gpu_endpoints[target_gpu.id]) {
          auto path =
              find_peer_path(*this, adjacency, unsafe_switches, source_endpoint, target_endpoint);
          if (path.empty())
            continue;
          if (!unique_path.empty() ||
              has_alternate_peer_path(*this, adjacency, unsafe_switches, source_endpoint,
                                      target_endpoint, path)) {
            ambiguous = true;
            break;
          }
          unique_path = std::move(path);
        }
        if (ambiguous)
          break;
      }
      if (unique_path.empty() || ambiguous || unique_path.size() > kMaxPeerRouteEndpoints)
        continue;

      SystemPeerRoute route;
      route.source_gpu = source_gpu.id;
      route.target_gpu = target_gpu.id;
      route.source_gpu_id = source_gpu.device.gpu_id;
      route.target_gpu_id = target_gpu.device.gpu_id;
      route.hop_classes.reserve(unique_path.size() - 1);
      for (size_t i = 1; i < unique_path.size(); ++i) {
        const std::string &a = endpoints[unique_path[i - 1]].id;
        const std::string &b = endpoints[unique_path[i]].id;
        const auto edge = std::find_if(links.begin(), links.end(), [&](const SystemLink &link) {
          return (link.src == a && link.dst == b) || (link.src == b && link.dst == a);
        });
        if (edge == links.end() || (edge->link_class != SystemLinkClass::ScaleUp &&
                                    edge->link_class != SystemLinkClass::InfinityFabric))
          break;
        route.hop_classes.push_back(edge->link_class);
      }
      if (route.hop_classes.size() != unique_path.size() - 1)
        continue;
      for (size_t endpoint : unique_path)
        route.endpoints.push_back(endpoints[endpoint].id);
      routes.push_back(std::move(route));
    }
  }
  return SystemPeerRouteProjection(selected.id, std::move(routes));
}

util::FailureOr<SystemGraph> system_graph_from_fb(const fb::SimulationConfig *config,
                                                  util::DiagnosticEmitter emit_error) {
  if (!config)
    return emit_error.emit() << "missing simulation config";
  return config->system() ? manifest_graph(config, emit_error) : legacy_graph(config, emit_error);
}

util::FailureOr<SystemGraph> load_system_graph(const std::string &json_path,
                                               const std::string &schema_text,
                                               util::DiagnosticEmitter emit_error) {
  auto json = read_config_file(json_path, emit_error);
  if (json.failed())
    return util::Result::failure();
  return load_system_graph_from_string(json.value(), schema_text, emit_error);
}

util::FailureOr<SystemGraph> load_system_graph_from_string(const std::string &json,
                                                           const std::string &schema_text,
                                                           util::DiagnosticEmitter emit_error) {
  bool has_manifest = false;
  auto graph = with_parsed_simulation_config_json(
      json, schema_text, emit_error,
      [&has_manifest, emit_error](const fb::SimulationConfig *config) {
        has_manifest = config->system() != nullptr;
        return system_graph_from_fb(config, emit_error);
      });
  if (graph.failed() || !has_manifest)
    return graph;
  // Preserve the legacy loader's forward-compatible unknown-field behavior,
  // but reject misspelled fields within an explicit system manifest.
  return with_parsed_simulation_config_json(
      json, schema_text, emit_error,
      [emit_error](const fb::SimulationConfig *config) {
        return system_graph_from_fb(config, emit_error);
      },
      false);
}

} // namespace rocjitsu::config
