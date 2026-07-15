// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file topology.h
/// @brief Simulation topology, adjacency graph, partitioning, and clock domain management.

#ifndef SIMDOJO_SIM_TOPOLOGY_H_
#define SIMDOJO_SIM_TOPOLOGY_H_

#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/exec_mode.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simdojo {

class Topology;

/// Specification for a deferred topology link (resolved from config patterns).
struct LinkSpec {
  std::string src; ///< Dotted path to source port.
  std::string dst; ///< Dotted path to destination port.
  uint32_t latency = 1;
  uint32_t weight = 1;
};

/// @brief A partition of the simulation topology.
///
/// @details Each Partition holds the set of components assigned to a single
/// worker thread, along with the links internal to and crossing this partition.
struct Partition {
  PartitionID id = 0;                  ///< Partition identifier.
  std::vector<Component *> components; ///< Components assigned to this partition.
  std::vector<Link *> internal_links;  ///< Links where both endpoints are in this partition.
  std::vector<Link *> boundary_links;  ///< Links crossing into or out of this partition.
  uint64_t total_weight = 0;           ///< Sum of component weights.
  std::unordered_set<PartitionID> neighbor_partitions; ///< Partitions connected via boundary links.
};

/// @brief Adjacency representation for the partitioning algorithm.
///
/// @details Converts the compound graph into a flat undirected graph where:
///   - Nodes = all components (leaves and composites)
///   - Edges = links between ports of different components
///   - Edge weights = link weights (for cut cost)
///   - Node weights = component weights (for balance)
class AdjacencyGraph {
public:
  /// @brief Build an adjacency graph from a collected component list and links.
  /// @param components Flat list of all components (from collect_all_components).
  /// @param links All links in the topology.
  /// @returns A flat undirected AdjacencyGraph.
  static AdjacencyGraph build(const std::vector<Component *> &components,
                              const std::vector<std::unique_ptr<Link>> &links);

  uint32_t num_nodes = 0; ///< Number of nodes (all components).
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>>
      adjacency;                      ///< Per-node adjacency list: (neighbor_index, edge_weight).
  std::vector<uint32_t> node_weights; ///< Per-node weights for balance constraints.
  std::vector<Component *> index_to_component; ///< Maps graph index back to the Component.
};

/// @brief The simulation compound graph.
///
/// @details Topology owns all components (via the root CompositeComponent), all links,
/// and all partitions. It provides component creation, link creation,
/// graph traversal, and partitioning into sub-graphs for parallel execution.
class Topology {
public:
  Topology() = default;

  /// @brief Set the root composite component (ownership transferred).
  /// @param root The root of the component tree.
  void set_root(std::unique_ptr<CompositeComponent> root) { root_ = std::move(root); }

  /// @brief Return the root composite component.
  /// @returns Pointer to the root, or nullptr if not set.
  CompositeComponent *root() const { return root_.get(); }

  /// @brief Create a link between two ports with a given latency.
  /// @param src Source port.
  /// @param dst Destination port.
  /// @param latency Propagation delay in simulation ticks.
  /// @param weight Partitioning cut weight for this link.
  /// @returns Raw pointer to the created link.
  Link *add_link(Port *src, Port *dst, Tick latency, uint32_t weight = 1);

  /// @brief Create two unidirectional links for bidirectional communication.
  ///
  /// @details Wires a_out -> b_in (forward, fwd_latency) and b_out -> a_in (reverse,
  /// rev_latency). Equivalent to calling add_link() twice.
  /// @param a_out Output port on component A.
  /// @param b_in Input port on component B.
  /// @param b_out Output port on component B.
  /// @param a_in Input port on component A.
  /// @param fwd_latency Latency from A to B.
  /// @param rev_latency Latency from B to A.
  /// @param weight Partitioning cut weight for both links.
  /// @returns Pair of raw pointers: {forward_link, reverse_link}.
  std::pair<Link *, Link *> add_bidirectional_link(Port *a_out, Port *b_in, Port *b_out, Port *a_in,
                                                   Tick fwd_latency, Tick rev_latency,
                                                   uint32_t weight = 1);

  /// @brief Return the list of all links.
  /// @returns Const reference to the link vector.
  const std::vector<std::unique_ptr<Link>> &links() const { return links_; }

  /// @brief Collect all components (including composites) into a flat vector.
  /// @returns Vector of pointers to all components in the tree.
  std::vector<Component *> collect_all_components() const;

  /// @brief Return the total number of components in the topology.
  /// @returns Component count.
  uint32_t num_components() const;

  /// @brief Return the list of partitions (const).
  /// @returns Const reference to the partition vector.
  const std::vector<Partition> &partitions() const { return partitions_; }

  /// @brief Return the list of partitions (mutable).
  /// @returns Mutable reference to the partition vector.
  std::vector<Partition> &partitions() { return partitions_; }

  /// @brief Perform DFS visiting each leaf component.
  /// @param visitor Callback invoked with (component, depth) for each leaf.
  void dfs_visit(std::function<void(Component *, uint32_t)> visitor) const;

  /// @brief Create and register a clock domain.
  /// @param name Human-readable domain name.
  /// @param frequency_hz Clock frequency in Hz.
  /// @param phase_offset Phase offset in simulation ticks.
  /// @returns Pointer to the created ClockDomain.
  ClockDomain *add_clock_domain(std::string name, uint64_t frequency_hz, Tick phase_offset = 0);

  /// @brief Return the list of registered clock domains.
  /// @returns Const reference to the clock domain vector.
  const std::vector<std::unique_ptr<ClockDomain>> &clock_domains() const { return clock_domains_; }

  /// @brief Partition the topology for parallel execution.
  /// @param num_partitions Number of partitions to create (one per thread).
  void partition(uint32_t num_partitions);

  /// @brief Resolve a dotted path (e.g., "xcd0.se1.cu3") to a Component.
  Component *resolve_component(const std::string &path) const;

  /// @brief Resolve a dotted path to a Port (last segment is the port name).
  Port *resolve_port(const std::string &path) const;

  /// @brief Wire deferred link specs, creating ports on demand.
  void wire_links(const std::vector<LinkSpec> &specs, ExecMode mode);

  /// @brief Partition the topology with explicit partition assignments.
  ///
  /// @details Bypasses the FM partitioner. Each component is assigned a
  /// partition ID by the caller's callback. Used by tests that need
  /// deterministic partition layouts.
  /// @param num_partitions Number of partitions to create.
  /// @param assigner Callback returning the partition ID for each component.
  void partition_manual(uint32_t num_partitions, std::function<PartitionID(Component *)> assigner);

private:
  /// @brief Classify links as internal or boundary for each partition.
  void classify_links();

  std::unique_ptr<CompositeComponent> root_;                ///< Root of the component tree.
  std::vector<std::unique_ptr<Link>> links_;                ///< All links in the topology.
  std::vector<std::unique_ptr<ClockDomain>> clock_domains_; ///< Registered clock domains.
  std::vector<Partition> partitions_;                       ///< Computed partitions.
  LinkID next_link_id_ = 0; ///< Counter for auto-assigning link IDs.
};

/// @brief Graph partitioner using a multilevel Fiduccia-Mattheyses (FM)
/// approach.
///
/// Self-contained implementation (no external libraries). The algorithm:
/// 1. Coarsening: heavy-edge matching to contract the graph
/// 2. Initial partitioning: FM bisection on the coarsened graph
/// 3. Uncoarsening/refinement: project back, FM refine at each level
/// 4. k-way: recursive bisection
class Partitioner {
public:
  /// @brief Configuration parameters for graph partitioning.
  struct Config {
    uint32_t num_partitions = 1;       ///< Target number of partitions.
    double imbalance_tolerance = 0.05; ///< Max allowed imbalance ratio (0.05 = 5%).
    uint32_t fm_max_passes = 10;       ///< Maximum FM refinement passes per level.
    uint32_t coarsen_threshold = 100;  ///< Stop coarsening when graph is this small.
  };

  /// @brief Construct a partitioner with the given configuration.
  /// @param config Partitioning parameters.
  explicit Partitioner(Config config) : config_(config) {}

  /// @brief Partition the graph and assign PartitionIDs to components.
  /// @param graph The adjacency graph to partition (components are updated in-place).
  /// @returns Vector of Partition descriptors.
  std::vector<Partition> partition(AdjacencyGraph &graph);

private:
  struct CoarsenLevel {
    AdjacencyGraph graph;
    std::vector<uint32_t> fine_to_coarse;
    std::vector<std::vector<uint32_t>> coarse_to_fine;
  };

  CoarsenLevel coarsen(const AdjacencyGraph &graph);

  void fm_bisect(const AdjacencyGraph &graph, std::vector<uint8_t> &assignment);

  int64_t fm_refine(const AdjacencyGraph &graph, std::vector<uint8_t> &assignment);

  void uncoarsen(const CoarsenLevel &level, const std::vector<uint8_t> &coarse_assignment,
                 std::vector<uint8_t> &fine_assignment);

  void recursive_bisect(AdjacencyGraph &graph, std::vector<PartitionID> &assignment,
                        PartitionID base_id, uint32_t num_parts);

  uint64_t compute_cut(const AdjacencyGraph &graph, const std::vector<uint8_t> &assignment) const;

  Config config_; ///< Partitioning parameters.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_TOPOLOGY_H_
