/* Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "hip_graph_internal.hpp"
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <iostream>

#define CASE_STRING(X, C)                                                                          \
  case X:                                                                                          \
    case_string = #C;                                                                              \
    break;
namespace {
const char* GetGraphNodeTypeString(uint32_t op) {
  const char* case_string;
  switch (static_cast<hipGraphNodeType>(op)) {
    CASE_STRING(hipGraphNodeTypeKernel, KernelNode)
    CASE_STRING(hipGraphNodeTypeMemcpy, MemcpyNode)
    CASE_STRING(hipGraphNodeTypeMemset, MemsetNode)
    CASE_STRING(hipGraphNodeTypeHost, HostNode)
    CASE_STRING(hipGraphNodeTypeGraph, GraphNode)
    CASE_STRING(hipGraphNodeTypeEmpty, EmptyNode)
    CASE_STRING(hipGraphNodeTypeWaitEvent, WaitEventNode)
    CASE_STRING(hipGraphNodeTypeEventRecord, EventRecordNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreSignal, ExtSemaphoreSignalNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreWait, ExtSemaphoreWaitNode)
    CASE_STRING(hipGraphNodeTypeMemAlloc, MemAllocNode)
    CASE_STRING(hipGraphNodeTypeMemFree, MemFreeNode)
    CASE_STRING(hipGraphNodeTypeMemcpyFromSymbol, MemcpyFromSymbolNode)
    CASE_STRING(hipGraphNodeTypeMemcpyToSymbol, MemcpyToSymbolNode)
    default:
      case_string = "Unknown node type";
  };
  return case_string;
};
}  // namespace

namespace hip {

int GraphNode::nextID = 0;
int Graph::nextID = 0;
std::unordered_set<GraphNode*> GraphNode::nodeSet_;
// Guards global node set
amd::Monitor GraphNode::nodeSetLock_{};
std::unordered_set<Graph*> Graph::graphSet_;
// Guards global graph set
amd::Monitor Graph::graphSetLock_{};
std::unordered_set<GraphExec*> GraphExec::graphExecSet_;
// Guards global exec graph set
// we have graphExec object as part of child graph and we need recursive lock
amd::Monitor GraphExec::graphExecSetLock_(true);
// Serialize the creation of internal streams from multiple threads, ensuring that each stream is
// mapped to different HSA queues.
amd::Monitor GraphExec::graphExecStreamCreateLock_(true);
std::unordered_set<UserObject*> UserObject::ObjectSet_;
// Guards global user object
amd::Monitor UserObject::UserObjectLock_{};
// Guards mem map add/remove against work thread
amd::Monitor GraphNode::WorkerThreadLock_{};

hipError_t GraphMemcpyNode1D::ValidateParams(void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  hipError_t status = ihipMemcpy_validate(dst, src, count, kind);
  if (status != hipSuccess) {
    return status;
  }
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);

  if ((srcMemory == nullptr) && (dstMemory != nullptr)) {  // host to device
    if ((kind != hipMemcpyHostToDevice) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {  // device to host
    if ((kind != hipMemcpyDeviceToHost) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t GraphMemcpyNode::ValidateParams(const hipMemcpy3DParms* pNodeParams) {
  hipError_t status;
  status = ihipMemcpy3D_validate(pNodeParams);
  if (status != hipSuccess) {
    return status;
  }

  const HIP_MEMCPY3D pCopy = hip::getDrvMemcpy3DDesc(*pNodeParams);
  status = ihipDrvMemcpy3D_validate(&pCopy);
  if (status != hipSuccess) {
    return status;
  }
  return hipSuccess;
}

// ================================================================================================
bool Graph::isGraphValid(Graph* pGraph) {
  amd::ScopedLock lock(graphSetLock_);
  if (graphSet_.find(pGraph) == graphSet_.end()) {
    return false;
  }
  return true;
}

// ================================================================================================
void Graph::AddNode(const Node& node) {
  vertices_.emplace_back(node);
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "[hipGraph] Add %s(%p)",
          GetGraphNodeTypeString(node->GetType()), node);
  node->SetParentGraph(this);
}

// ================================================================================================
void Graph::RemoveNode(const Node& node) {
  vertices_.erase(std::remove(vertices_.begin(), vertices_.end(), node), vertices_.end());
  delete node;
}

// ================================================================================================
std::vector<Node> Graph::GetRootNodes() const {
  // root nodes are all vertices with 0 in-degrees
  std::vector<Node> roots;

  for (const auto& entry : vertices_) {
    if (entry->GetInDegree() == 0) {
      roots.push_back(entry);
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "[hipGraph] Root node: %s(%p)",
              GetGraphNodeTypeString(entry->GetType()), entry);
    }
  }
  // Use move semantics for efficient return
  return std::move(roots);
}

// ================================================================================================
// leaf nodes are all vertices with 0 out-degrees
std::vector<Node> Graph::GetLeafNodes() const {
  std::vector<Node> leafNodes;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      leafNodes.push_back(entry);
    }
  }
  return leafNodes;
}

// ================================================================================================
size_t Graph::GetLeafNodeCount() const {
  int numLeafNodes = 0;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      numLeafNodes++;
    }
  }
  return numLeafNodes;
}

std::vector<std::pair<Node, Node>> Graph::GetEdges() const {
  std::vector<std::pair<Node, Node>> edges;
  for (const auto& i : vertices_) {
    for (const auto& j : i->GetEdges()) {
      edges.push_back(std::make_pair(i, j));
    }
  }
  return edges;
}

// ================================================================================================
void Graph::ScheduleOneNode(Node node, int stream_id) {
  if (node->stream_id_ == -1) {
    // Assign active stream to the current node
    node->stream_id_ = stream_id;
    max_streams_ = std::max(max_streams_, (stream_id + 1));
    // Track which devices are used by each stream for multi-device graph execution
    streams_dev_ids_[stream_id].insert(node->dev_id_);
    // Process child graph separately, since, there is no connection
    if (node->GetType() == hipGraphNodeTypeGraph) {
      auto child = reinterpret_cast<hip::ChildGraphNode*>(node)->GetChildGraph();
      hipError_t status = child->ScheduleNodes();
      max_streams_ = std::max(max_streams_, child->max_streams_);
      if (child->max_streams_ == 1) {
        reinterpret_cast<hip::ChildGraphNode*>(node)->GraphExec::TopologicalOrder();
      }
    }
    for (auto edge : node->GetEdges()) {
      ScheduleOneNode(edge, stream_id);
      // 1. Each extra edge will get a new stream from the pool
      // 2. Streams will be reused if the number of edges > streams
      stream_id = (stream_id + 1) % DEBUG_HIP_FORCE_GRAPH_QUEUES;
    }
  }
}

// ================================================================================================
hipError_t Graph::ScheduleNodes() {
  // Set graphExec_ if this is called on a GraphExec object
  // dynamic_cast returns nullptr if 'this' is not a GraphExec
  graphExec_ = dynamic_cast<GraphExec*>(this);

  if (DEBUG_HIP_GRAPH_PACKET_ENGINE) {
    // New batch-based scheduling logic
    return ScheduleNodesIntoBatches();
  } else {
    // Classic scheduling logic
  memset(&roots_[0], 0, sizeof(Node) * roots_.size());
  max_streams_ = 0;

  int stream_id = 0;
  for (auto node : vertices_) {
    if (node->stream_id_ == -1) {
      ScheduleOneNode(node, stream_id);
      // Find the root nodes
      if ((node->GetDependencies().size() == 0) && (node->stream_id_ != 0)) {
        // Fill in only the first in the sequence
        if (roots_[node->stream_id_] == nullptr) {
          roots_[node->stream_id_] = node;
        }
      }
      // 1. Each extra root will get a new stream from the pool
      // 2. Streams will be recycled if the number of roots > streams
      stream_id = (stream_id + 1) % DEBUG_HIP_FORCE_GRAPH_QUEUES;
    }
    }

    // Topological order is only needed for original scheduling
    if (graphExec_ && !graphExec_->TopologicalOrder()) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] TopologicalOrder failed - invalid graph");
      return hipErrorInvalidValue;
    }

    return hipSuccess;
  }
}

// ================================================================================================
hipError_t Graph::ScheduleNodesIntoBatches() {
  // Find execution paths and create segments
  auto execution_paths = FindExecutionPaths();

  if (execution_paths.empty()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] No execution paths found in graph");
    return hipErrorInvalidValue;
  }

  CreateSegmentsFromPaths(execution_paths);
  // Resolve segment dependencies and calculate dependency levels
  ResolveSegmentDependencies();

  return hipSuccess;
}

// ================================================================================================
// Flatten child graph metadata nodes in a path
// Metadata nodes can be at the beginning (for dependencies) or end (for edges)
void Graph::FlattenChildGraphMetadata(std::vector<Node>& path) {
  if (path.empty()) {
    return;
  }

  // As per the logic in FindPathsRecursive we prepend metadata nodes for dependencies
  // and append metadata nodes for edges
  bool has_metadata_at_front = (path[0]->GetType() == hipGraphNodeTypeGraph);
  bool has_metadata_at_back = (path.back()->GetType() == hipGraphNodeTypeGraph);

  if (!has_metadata_at_front && !has_metadata_at_back) {
    return; // No metadata to flatten
  }

  // Handle dependencies
  if (has_metadata_at_front && path.size() > 1) {
    // Find the first real (non-child-graph) node after metadata nodes
    Node first_real_node = nullptr;
    size_t first_real_idx = 0;
    for (size_t i = 1; i < path.size(); ++i) {
      if (path[i]->GetType() != hipGraphNodeTypeGraph) {
        first_real_node = path[i];
        first_real_idx = i;
        break;
      }
    }

    // Transfer dependencies: Only if first real node has no original dependencies
    if (first_real_node != nullptr) {
      auto first_deps = first_real_node->GetDependencies();
      if (first_deps.empty()) {
        // Collect all dependencies from preceding child graph nodes
        for (size_t i = 0; i < first_real_idx; ++i) {
          Node child_graph_node = path[i];
          const auto& cg_deps = child_graph_node->GetDependencies();

          for (const auto& dep : cg_deps) {
            first_deps.push_back(dep);

            // Update reverse pointer: dep's edge should point to first_real_node
            auto dep_edges = dep->GetEdges();
            for (auto& edge : dep_edges) {
              if (edge == child_graph_node) {
                edge = first_real_node;
              }
            }
            dep->SetEdges(dep_edges);
          }
        }
        first_real_node->SetDependencies(first_deps);
      }
    }
  }

  // Handle edges
  if (has_metadata_at_back && path.size() > 1) {
    // Find the last real (non-child-graph) node before metadata nodes
    Node last_real_node = nullptr;
    int last_real_idx = -1;
    for (int i = path.size() - 1; i >= 0; --i) {
      if (path[i]->GetType() != hipGraphNodeTypeGraph) {
        last_real_node = path[i];
        last_real_idx = i;
        break;
      }
    }

    // Transfer edges: Only if last real node has no original edges
    if (last_real_node != nullptr) {
      auto last_edges = last_real_node->GetEdges();
      if (last_edges.empty()) {
        // Collect all edges from child graph nodes after last_real_node
        for (size_t i = last_real_idx + 1; i < path.size(); ++i) {
          Node child_graph_node = path[i];
          const auto& cg_edges = child_graph_node->GetEdges();

          for (const auto& edge : cg_edges) {
            last_edges.push_back(edge);

            // Update reverse pointer: edge's dependency should point to last_real_node
            auto edge_deps = edge->GetDependencies();
            for (auto& dep : edge_deps) {
              if (dep == child_graph_node) {
                dep = last_real_node;
              }
            }
            edge->SetDependencies(edge_deps);
          }
        }
        last_real_node->SetEdges(last_edges);
      }
    }
  }
}

// ================================================================================================
void Graph::CreateSegmentsFromPaths(const std::vector<std::vector<Node>>& execution_paths) {
  // Access segments from GraphExec if set, otherwise use Graph's members
  auto& segments = graphExec_->segments_;
  auto& node_to_segment_id = graphExec_->node_to_segment_id_;

  // Clear previous segments
  segments.clear();
  node_to_segment_id.clear();

  // Create a segment for each execution path
  for (size_t i = 0; i < execution_paths.size(); ++i) {
    const auto& path = execution_paths[i];
    if (path.empty()) continue;

    Segment segment;
    segment.id = i;
    segment.nodes = path;
    segment.first_node = path.front();
    segment.last_node = path.back();

    // Handle child graph metadata nodes (can be at beginning, end, or both)
    bool has_metadata = (!path.empty() && path[0]->GetType() == hipGraphNodeTypeGraph) ||
                        (!path.empty() && path.back()->GetType() == hipGraphNodeTypeGraph);

    if (has_metadata) {
      // Flatten child graph metadata - transfer dependencies/edges through the path
      FlattenChildGraphMetadata(segment.nodes);

      // Remove metadata nodes from front
      while (!segment.nodes.empty() && segment.nodes.front()->GetType() == hipGraphNodeTypeGraph) {
        segment.nodes.erase(segment.nodes.begin());
      }

      // Remove metadata nodes from back
      while (!segment.nodes.empty() && segment.nodes.back()->GetType() == hipGraphNodeTypeGraph) {
        segment.nodes.pop_back();
      }

      // Update first/last pointers
      if (!segment.nodes.empty()) {
        segment.first_node = segment.nodes.front();
        segment.last_node = segment.nodes.back();
      }
    }

    segments.push_back(segment);

    // Map each node in this segment to the segment ID (excluding metadata node)
    for (const auto& node : segment.nodes) {
      node_to_segment_id[node] = i;
    }
  }
}

// ================================================================================================
void Graph::ResolveSegmentDependencies() {
  // Access segments from GraphExec if set, otherwise use Graph's members
  auto& segments = graphExec_->segments_;
  auto& node_to_segment_id = graphExec_->node_to_segment_id_;

  // Resolve segment dependencies
  for (size_t i = 0; i < segments.size(); ++i) {
    auto& segment = segments[i];

    // Only check first node for incoming dependencies
    if (segment.first_node != nullptr) {
      const auto& dependencies = segment.first_node->GetDependencies();

      for (const auto& dep_node : dependencies) {
        // Find which segment this dependency belongs to
        auto dep_it = node_to_segment_id.find(dep_node);
        if (dep_it != node_to_segment_id.end()) {
          int dep_segment_id = dep_it->second;

          // Add dependency if not already present
          if (std::find(segment.segment_ids_dependencies.begin(),
                       segment.segment_ids_dependencies.end(),
                       dep_segment_id) == segment.segment_ids_dependencies.end()) {
            segment.segment_ids_dependencies.push_back(dep_segment_id);

            // Also add this segment as an edge of the dependency segment
            segments[dep_segment_id].segment_ids_edges.push_back(i);
          }
        }
      }
    }
  }

  // Calculate dependency levels using topological sort
  // This also calculates max_streams_ based on maximum parallelism
  CalculateSegmentTopoDependencyLevels();
}

// ================================================================================================
void Graph::CalculateSegmentTopoDependencyLevels() {
  // segments_ and segments_per_level_ are from GraphExec
  auto& segments = graphExec_->segments_;
  auto& max_dependency_level = graphExec_->max_dependency_level_;
  auto& segments_per_level = graphExec_->segments_per_level_;

  // Topological sort of segments to calculate dependency levels
  // Assume each segment is a node and the dependencies are segments edges
  // Segments with same dependency level can be processed in parallel
  std::queue<int> queue;
  std::unordered_map<int, int> in_degree;

  // Reset max dependency level, max streams, and segments per level
  max_dependency_level = -1;
  max_streams_ = 1;
  segments_per_level.clear();

  // Initialize in-degree for each segment and enqueue root segments
  for (size_t i = 0; i < segments.size(); ++i) {
    segments[i].dependency_level = -1;
    in_degree[i] = segments[i].segment_ids_dependencies.size();

    if (in_degree[i] == 0) {
      // Root segments have level 0
      segments[i].dependency_level = 0;
      queue.push(i);
      max_dependency_level = 0;
      segments_per_level[0].push_back(i);
    }
  }

  // Process segments in topological order
  while (!queue.empty()) {
    int current_id = queue.front();
    queue.pop();

    auto& current_segment = segments[current_id];
    int current_level = current_segment.dependency_level;

    // Process all segments that depend on current segment
    for (int edge_id : current_segment.segment_ids_edges) {
      auto& edge_segment = segments[edge_id];

      // Calculate the dependency level for this segment
      // It's one level higher than the maximum of its dependencies
      int new_level = current_level + 1;
      if (edge_segment.dependency_level < new_level) {
        edge_segment.dependency_level = new_level;
        // Track the maximum dependency level
        max_dependency_level = std::max(max_dependency_level, new_level);
      }

      // Decrease in-degree and enqueue if all dependencies processed
      in_degree[edge_id]--;
      if (in_degree[edge_id] == 0) {
        queue.push(edge_id);
        // Add segment to its dependency level
        segments_per_level[edge_segment.dependency_level].push_back(edge_id);
      }
    }
  }

  // Calculate max_streams_ based on maximum parallelism at any dependency level
  for (const auto& level_segments : segments_per_level) {
    max_streams_ = std::max(max_streams_, static_cast<int>(level_segments.second.size()));
  }

  // If there's only 1 segment total, use single stream
  if (segments.size() == 1) {
    max_streams_ = 1;
  }
}

// ================================================================================================
std::vector<std::vector<Node>> Graph::FindExecutionPaths() {
  std::vector<std::vector<Node>> all_paths;

  // Find all root nodes (nodes with no dependencies)
  const auto& root_nodes = GetRootNodes();

  std::unordered_set<unsigned int> visited;
  for (const auto& root : root_nodes) {
    // For each root, find all possible paths starting from it
    // Each root gets its own visited set to allow full path expansion
    std::vector<Node> current_path;
    FindPathsRecursive(root, current_path, visited, all_paths);
  }

  return all_paths;
}

// ================================================================================================
void Graph::FindPathsRecursive(Node node, std::vector<Node>& current_path,
                               std::unordered_set<unsigned int>& visited,
                               std::vector<std::vector<Node>>& all_paths) {
  // Check if already visited
  if (visited.find(node->GetID()) != visited.end()) {
    return;
  }

  // Mark regular nodes as visited
  visited.insert(node->GetID());

  // Check if device ID changed from previous node in path
  bool device_changed = false;
  if (!current_path.empty()) {
    int prev_device_id = current_path.back()->GetDeviceId();
    int curr_device_id = node->GetDeviceId();
    if (prev_device_id != curr_device_id) {
      device_changed = true;
      // Save current path before device change
      all_paths.push_back(current_path);
      current_path.clear();
    }
  }

  // Handle child graph nodes specially
  if (node->GetType() == hipGraphNodeTypeGraph) {

    // Save path before child graph node
    if (!current_path.empty()) {
      all_paths.push_back(current_path);
    }

    // Get the child graph and recursively process it
    auto childGraphNode = reinterpret_cast<hip::ChildGraphNode*>(node);
    auto childGraph = childGraphNode->GetChildGraph();

    if (childGraph != nullptr) {
      // Find all paths in the child graph
      std::vector<std::vector<Node>> child_paths = childGraph->FindExecutionPaths();

      // For each child path, conditionally add parent node as metadata
      for (auto& child_path : child_paths) {
        if (!child_path.empty()) {
          Node first_node = child_path.front();
          Node last_node = child_path.back();

          // Check if we need to prepend/append metadata node
          bool prepend_metadata = first_node->GetDependencies().empty();
          bool append_metadata = last_node->GetEdges().empty();

          std::vector<Node> combined_path;

          // Prepend metadata node if first node is a root
          // Will transfer dependencies from parent to child
          if (prepend_metadata) {
            combined_path.push_back(node);
          }

          // Add the child path
          combined_path.insert(combined_path.end(), child_path.begin(), child_path.end());

          // Append metadata node if last node is a leaf
          // Will transfer edges from parent to child
          if (append_metadata) {
            combined_path.push_back(node);
          }

          all_paths.push_back(std::move(combined_path));
        }
      }
    }

    // Clear current path and continue with edges from the child graph node
    current_path.clear();
    const auto& edges = node->GetEdges();
    for (const auto& edge : edges) {
      FindPathsRecursive(edge, current_path, visited, all_paths);
    }

    return;
  }

  current_path.push_back(node);

  // Edges are out degrees, Dependencies are in degrees
  const auto& edges = node->GetEdges();
  const auto& dependencies = node->GetDependencies();

  // Check if this is a fork node (multiple outgoing edges)
  bool is_fork = edges.size() > 1;
  // Check if this is a join node (multiple incoming dependencies)
  bool is_join = dependencies.size() > 1;

  if (is_fork || is_join) {
    // Save current path as a separate segment
    if (!current_path.empty()) {
      std::vector<Node> path_to_save = current_path;
      Node saved_join_node = nullptr;

      // For join nodes, save path without the join node itself
      // For fork nodes, save the complete path
      if (is_join) {
        saved_join_node = path_to_save.back();
        path_to_save.pop_back();
      }

      all_paths.push_back(std::move(path_to_save));
      current_path.clear();

      // Put the join node back in current_path for further traversal
      if (saved_join_node != nullptr) {
        current_path.push_back(saved_join_node);
      }
    }

    // Traverse each branch until it hits a join
    for (const auto& edge : edges) {
      FindPathsRecursive(edge, current_path, visited, all_paths);

      // Save the path if it's not empty and this was a fork/join boundary
      if (!current_path.empty() && (is_fork || is_join)) {
        all_paths.push_back(current_path);
        current_path.clear();
      }
    }

  } else if (edges.size() == 1) {
    // Single edge - continue on same path
    FindPathsRecursive(edges[0], current_path, visited, all_paths);
  }

  // Save any remaining path (handles leaf nodes and leaf join nodes)
  if (!current_path.empty()) {
    all_paths.push_back(current_path);
    current_path.clear();
  }
}


// ================================================================================================
bool Graph::TopologicalOrder(std::vector<Node>& TopoOrder) {
  std::queue<Node> q;
  std::unordered_map<Node, int> inDegree;
  for (auto entry : vertices_) {
    // Update the dependencies if a signal is required
    for (auto dep : entry->GetDependencies()) {
      // Check if the stream ID doesn't match and enable signal
      if (dep->stream_id_ != entry->stream_id_) {
        dep->signal_is_required_ = true;
      }
    }

    if (entry->GetInDegree() == 0) {
      q.push(entry);
    }
    inDegree[entry] = entry->GetInDegree();
  }
  while (!q.empty()) {
    Node node = q.front();
    TopoOrder.push_back(node);
    q.pop();
    for (auto edge : node->GetEdges()) {
      inDegree[edge]--;
      if (inDegree[edge] == 0) {
        q.push(edge);
      }
    }
  }
  if (GetNodeCount() == TopoOrder.size()) {
    return true;
  }
  return false;
}

// ================================================================================================
void Graph::clone(Graph* newGraph, bool cloneNodes) const {
  newGraph->pOriginalGraph_ = this;
  for (hip::GraphNode* entry : vertices_) {
    GraphNode* node = entry->clone();
    node->SetParentGraph(newGraph);
    newGraph->vertices_.push_back(node);
    newGraph->clonedNodes_[entry] = node;
  }

  std::vector<Node> clonedEdges;
  std::vector<Node> clonedDependencies;
  for (auto node : vertices_) {
    const std::vector<Node>& edges = node->GetEdges();
    clonedEdges.clear();
    for (auto edge : edges) {
      clonedEdges.push_back(newGraph->clonedNodes_[edge]);
    }
    newGraph->clonedNodes_[node]->SetEdges(clonedEdges);
  }
  for (auto node : vertices_) {
    const std::vector<Node>& dependencies = node->GetDependencies();
    clonedDependencies.clear();
    for (auto dep : dependencies) {
      clonedDependencies.push_back(newGraph->clonedNodes_[dep]);
    }
    newGraph->clonedNodes_[node]->SetDependencies(clonedDependencies);
  }
  for (auto& userObj : graphUserObj_) {
    userObj.first->retain();
    newGraph->graphUserObj_.insert(userObj);
    // Clone graph should have its separate graph owned ref count = 1
    newGraph->graphUserObj_[userObj.first] = 1;
    userObj.first->owning_graphs_.insert(newGraph);
  }
  // Clone the root nodes to the new graph
  if (roots_.size() > 0) {
    memcpy(&newGraph->roots_[0], &roots_[0], sizeof(Node) * roots_.size());
  }
  newGraph->memAllocNodePtrs_ = memAllocNodePtrs_;
  if (!cloneNodes) {
    newGraph->clonedNodes_.clear();
  }
}

// ================================================================================================
Graph* Graph::clone() const {
  Graph* newGraph = new Graph(getCurrentDevice());
  clone(newGraph);
  return newGraph;
}

// ================================================================================================
bool GraphExec::isGraphExecValid(GraphExec* pGraphExec) {
  amd::ScopedLock lock(graphExecSetLock_);
  if (graphExecSet_.find(pGraphExec) == graphExecSet_.end()) {
    return false;
  }
  return true;
}

// ================================================================================================
hipError_t GraphExec::CreateStreams(uint32_t num_streams, int devId) {
  amd::ScopedLock lock(graphExecStreamCreateLock_);

  if (num_streams == 0) {
    ClPrint(amd::LOG_WARNING, amd::LOG_CODE,
            "[hipGraph] Attempting to create 0 streams for device %d", devId);
    return hipSuccess;
  }

  if (devId < 0 || devId >= g_devices.size() || g_devices[devId] == nullptr) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Invalid device ID %d for stream creation",
            devId);
    return hipErrorInvalidDevice;
  }

  // Check if streams already exist for this device
  if (parallel_streams_.find(devId) != parallel_streams_.end() &&
      !parallel_streams_[devId].empty()) {
    ClPrint(amd::LOG_WARNING, amd::LOG_CODE,
            "[hipGraph] Streams already exist for device %d, skipping creation", devId);
    return hipSuccess;
  }

  // Cap the number of streams to DEBUG_HIP_FORCE_GRAPH_QUEUES
  uint32_t max_streams = std::min(num_streams, DEBUG_HIP_FORCE_GRAPH_QUEUES);
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Creating %u parallel streams for device %d",
    max_streams, devId);
  parallel_streams_[devId].reserve(max_streams);
  for (uint32_t i = 0; i < max_streams; ++i) {
    auto stream = new hip::Stream(hip::getCurrentDevice(), hip::Stream::Priority::Normal,
                                  hipStreamNonBlocking);

    if (stream == nullptr || !stream->Create()) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to %s stream %u for device %d",
              stream == nullptr ? "allocate" : "create", i, devId);
      if (stream != nullptr) {
        hip::Stream::Destroy(stream);
      }
      // Clean up any previously created streams for this device
      for (auto& created_stream : parallel_streams_[devId]) {
        hip::Stream::Destroy(created_stream);
      }
      parallel_streams_[devId].clear();
      return hipErrorOutOfMemory;
    }

    parallel_streams_[devId].push_back(stream);
  }
  return hipSuccess;
}

// ================================================================================================
void GraphExec::FindStreamsReqPerDev() {
  // Count streams required per device based on stream-to-device mappings
  for (auto const& [stream_id, dev_ids] : streams_dev_ids_) {
    for (auto dev_id : dev_ids) {
      max_streams_dev_[dev_id]++;
    }
  }

  // Recursively process child graphs to determine their stream requirements
  for (auto node : vertices_) {
    if (node->GetType() == hipGraphNodeTypeGraph) {
      auto childNode = reinterpret_cast<ChildGraphNode*>(node);

      // Recursively find stream requirements for child graph
      childNode->FindStreamsReqPerDev();

      // Merge child graph's stream requirements with parent graph
      // Take the maximum streams needed per device to handle concurrent execution
      for (auto const& [dev_id, num_streams] : childNode->max_streams_dev_) {
        auto it = max_streams_dev_.find(dev_id);
        if (it != max_streams_dev_.end()) {
          // Device already has stream requirements - take the maximum
          max_streams_dev_[dev_id] = std::max(max_streams_dev_[dev_id], num_streams);
        } else {
          // New device - initialize with child graph's requirement
          max_streams_dev_[dev_id] = num_streams;
        }
      }
    }
  }
}

// ================================================================================================
hipError_t GraphExec::Init() {
  hipError_t status = hipSuccess;
  // create extra stream to avoid queue collision with the default execution stream
  if (max_streams_ > 1) {
    FindStreamsReqPerDev();
    if (max_streams_dev_.size() > 1) {
      // Multi-device graph detected - create parallel streams for each device
      for (auto const& [dev_id, num_streams] : max_streams_dev_) {
        ClPrint(amd::LOG_INFO, amd::LOG_API,
                "[hipGraph] For device id :%d max streams :%d for execution.\n", dev_id,
                num_streams);
        status = CreateStreams(num_streams, dev_id);
        if (status != hipSuccess) {
          return status;
        }
      }
    }
    status = CreateStreams(max_streams_, hip::getCurrentDevice()->deviceId());
  }
  if (status != hipSuccess) {
    return status;
  }
  if (DEBUG_HIP_GRAPH_PACKET_ENGINE) {
    // For graph nodes capture AQL packets to dispatch them directly during graph launch.
    status = CaptureAQLPackets();
  }

  instantiateDeviceId_ = hip::getCurrentDevice()->deviceId();
  static_cast<ReferenceCountedObject*>(hip::getCurrentDevice())->retain();
  return status;
}

//! Chunk size to add to kern arg pool
constexpr uint32_t kKernArgChunkSize = 128 * Ki;
// ================================================================================================
void GraphExec::GetKernelArgSizeForGraph(std::unordered_map<int, size_t>& kernArgSizeForGraph) {
  // Calculate the kernel argument size required for all graph kernel nodes
  // when GPU packet capture is enabled

  if (DEBUG_HIP_GRAPH_PACKET_ENGINE && !segments_.empty()) {
    for (const auto& segment : segments_) {
      for (hip::GraphNode* node : segment.nodes) {
        if (node->GraphCaptureEnabled()) {
          // Accumulate the kernel argument size for each device
          kernArgSizeForGraph[node->dev_id_] += node->GetKerArgSize();
        }
      }
    }
  }
}
// ================================================================================================
// Enable or disable a graph node's packets in the batch
// Simply updates the enabled state and count of disabled nodes
// ================================================================================================
void GraphExec::PacketBatch::setEnabled(GraphNode* node, bool enabled) {
  auto it = nodeToRangeIndex.find(node);
  if (it == nodeToRangeIndex.end()) {
    return;
  }
  NodeRange& range = nodeRanges[it->second];
  // Early return if state hasn't changed
  if (range.enabled == enabled) {
    return;
  }
  // Update counter based on state change
  if (enabled) {
    // Node being enabled: decrement counter
    disabledNodeCount--;
  } else {
    // Node being disabled: increment counter
    disabledNodeCount++;
  }
  range.enabled = enabled;
}

// ================================================================================================
// Rebuild cached filtered lists of enabled packets
// Only rebuilds if cache is stale (size doesn't match expected enabled count)
// ================================================================================================
void GraphExec::PacketBatch::rebuildFilteredLists() {
  // Calculate expected size based on currently enabled nodes
  size_t expectedCount = 0;
  for (const auto& range : nodeRanges) {
    if (range.enabled) {
      expectedCount += range.packetCount;
    }
  }

  // Cache is valid if size matches - no rebuild needed
  if (enabledPackets.size() == expectedCount) {
    return;
  }

  // Cache is stale - rebuild it
  enabledPackets.clear();
  enabledKernelNames.clear();

  enabledPackets.reserve(expectedCount);
  enabledKernelNames.reserve(expectedCount);

  // Build filtered lists from enabled node ranges
  for (const auto& range : nodeRanges) {
    if (range.enabled) {
      for (size_t j = 0; j < range.packetCount; ++j) {
        size_t packetIndex = range.startIndex + j;
        enabledPackets.push_back(dispatchPackets[packetIndex]);
        enabledKernelNames.push_back(dispatchKernelNames[packetIndex]);
      }
    }
  }
}

// ================================================================================================
hipError_t GraphExec::CaptureAndFormPacketsForGraph() {
  // Fixme: Only single stream child graph nodes are supported.
  hipError_t status = hipSuccess;

  // Clear previous batches
  segmentBatches_.clear();

  // Process nodes from segments
  for (const auto& segment : segments_) {
    // Create a SegmentBatch for this segment
    segmentBatches_.emplace_back(segment.id);
    // Initialize node_capture_status for this segment
    auto& currentSegBatch = segmentBatches_.back();
    currentSegBatch.node_capture_status.resize(segment.nodes.size(), false);
    for (size_t i = 0; i < segment.nodes.size(); ++i) {
      auto& node = segment.nodes[i];

      // Check if kernel node requires hidden heap and set it for the entire graph
      if (node->GetType() == hipGraphNodeTypeKernel) {
        static bool initialized = false;
        if (!initialized && reinterpret_cast<hip::GraphKernelNode*>(node)->HasHiddenHeap()) {
          SetHiddenHeap();
          initialized = true;
        }
      }

      // Handle nodes that support graph capture
      if (node->GraphCaptureEnabled()) {
        // Start of a new batch
        PacketBatch newBatch;

        // Collect packets from consecutive captured nodes
        size_t j = i;
        while (j < segment.nodes.size() && segment.nodes[j]->GraphCaptureEnabled()) {
          auto& currentNode = segment.nodes[j];
          // Capture packets for this node
          std::vector<uint8_t*> nodePackets;
          std::vector<std::string> nodeKernelNames;
          status = currentNode->CaptureAndFormPacket(GetKernelArgManager(), &nodePackets,
                                                     &nodeKernelNames);

          if (status != hipSuccess || nodePackets.empty()) {
            LogError("Packet capture failed");
            return status;
          }

          // Create NodeRange for this node
          // RangeIndex is 0 at the start
          const size_t rangeIndex = newBatch.nodeRanges.size();
          const size_t startIndex = newBatch.dispatchPackets.size();
          const size_t packetCount = nodePackets.size();

          // Reserve space to avoid reallocations during insertion
          newBatch.dispatchPackets.reserve(startIndex + packetCount);
          newBatch.dispatchKernelNames.reserve(startIndex + packetCount);

          // Add to dispatch lists (initially all enabled)
          newBatch.dispatchPackets.insert(newBatch.dispatchPackets.end(), nodePackets.begin(),
                                          nodePackets.end());
          newBatch.dispatchKernelNames.insert(newBatch.dispatchKernelNames.end(),
                                              nodeKernelNames.begin(), nodeKernelNames.end());

          // Store node mapping with range info
          newBatch.nodeRanges.push_back({startIndex, packetCount, true});
          newBatch.nodeToRangeIndex[currentNode] = rangeIndex;

          // Mark this node as successfully captured
          currentSegBatch.node_capture_status[j] = true;
          ++j;
        }

        // Add the batch if it has packets
        if (!newBatch.dispatchPackets.empty()) {
          currentSegBatch.packet_batches.emplace_back(std::move(newBatch));
        }

        // Skip the nodes we just processed, the index will be incremented by the loop
        i = j - 1;
      } else if (node->GetType() == hipGraphNodeTypeGraph) {
        auto childNode = reinterpret_cast<hip::ChildGraphNode*>(node);
        if (childNode->GetChildGraph()->max_streams_ == 1) {
          childNode->SetGraphCaptureStatus(true);
          status = childNode->CaptureAndFormPacketsForGraph();
          currentSegBatch.node_capture_status[i] = (status == hipSuccess);
          if (status != hipSuccess) {
            LogWarning("Child graph packet capture failed continuing with other nodes");
            status = hipSuccess;  // Continue processing other nodes
          }
        }
      }
    }
  }
  return status;
}

// ================================================================================================
hipError_t GraphExec::CaptureAQLPackets() {
  hipError_t status = hipSuccess;

  // Create a map to track kernel argument sizes for each device
  std::unordered_map<int, size_t> kernArgSizeForGraph;
  // Reserve space for all available devices and Initialize to 0
  kernArgSizeForGraph.reserve(g_devices.size());
  for (int devId = 0; devId < g_devices.size(); devId++) {
    kernArgSizeForGraph[devId] = 0;
  }
  GetKernelArgSizeForGraph(kernArgSizeForGraph);

  // Allocate kernel argument pools on respective devices with extra space for updates
  for (const auto& deviceKernArgPair : kernArgSizeForGraph) {
    const int deviceId = deviceKernArgPair.first;
    const size_t kernArgSize = deviceKernArgPair.second;

    if (kernArgSize == 0) {
      continue;
    }

    const size_t totalPoolSize = kernArgSize + kKernArgChunkSize;
    if (!kernArgManager_->AllocGraphKernargPool(totalPoolSize, g_devices[deviceId]->devices()[0])) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
              "[hipGraph] Failed to allocate kernel argument pool of size %zu for device %d",
              totalPoolSize, deviceId);
    return hipErrorMemoryAllocation;
    }
  }

  status = CaptureAndFormPacketsForGraph();
  if (status != hipSuccess) {
    return status;
  }

  kernArgManager_->ReadBackOrFlush();
  return hipSuccess;;
}

// ================================================================================================
hipError_t GraphExec::UpdateAQLPacket(hip::GraphNode* node) {
  if (!node->GraphCaptureEnabled()) {
    return hipSuccess;
  }
  // Todo: Add batching support for multi-device linear graph
  if (max_streams_dev_.size() == 1) {
    // Use node_to_segment_id_ for O(1) segment lookup
    auto segIdIt = node_to_segment_id_.find(node);
    if (segIdIt == node_to_segment_id_.end()) {
      return hipSuccess; // Node not in any segment
    }

    int segmentId = segIdIt->second;

    // Find the segment batch for this segment ID
    for (auto& segBatch : segmentBatches_) {
      if (segBatch.segment_id != segmentId) {
        continue;
      }

      // Search only within this segment's packet batches
      for (auto& packetBatch : segBatch.packet_batches) {
        auto it = packetBatch.nodeToRangeIndex.find(node);
        if (it != packetBatch.nodeToRangeIndex.end()) {
          // Found the batch containing this node - update packets
          PacketBatch::NodeRange& range = packetBatch.nodeRanges[it->second];

          // Capture new packets for this node
          std::vector<uint8_t*> newPackets;
          std::vector<std::string> newKernelNames;
          hipError_t status =
              node->CaptureAndFormPacket(kernArgManager_, &newPackets, &newKernelNames);
          if (status != hipSuccess) {
            return status;
          }
          // Number of packets per node can change
          const size_t oldPacketCount = range.packetCount;
          const size_t newPacketCount = newPackets.size();

          if (newPacketCount != oldPacketCount) {
            const size_t rangeIdx = it->second;
            const int64_t packetDelta = static_cast<int64_t>(newPacketCount) -
                                        static_cast<int64_t>(oldPacketCount);

            ClPrint(
                amd::LOG_INFO, amd::LOG_CODE,
                "[hipGraph] Packet count change for node (type=%d): %zu -> %zu packets (delta=%ld)",
                node->GetType(), oldPacketCount, newPacketCount, packetDelta);

            if (packetDelta > 0) {
              // Insert additional packet slots at the end of this node's range
              const size_t insertPos = range.startIndex + oldPacketCount;
              packetBatch.dispatchPackets.insert(packetBatch.dispatchPackets.begin() + insertPos,
                                                static_cast<size_t>(packetDelta), nullptr);
              packetBatch.dispatchKernelNames.insert(
                  packetBatch.dispatchKernelNames.begin() + insertPos,
                  static_cast<size_t>(packetDelta), std::string());
            } else {
              // Negative packetDelta, remove excess packet slots from the end of this node's range
              const size_t removePos = range.startIndex + newPacketCount;
              const size_t removeCount = oldPacketCount - newPacketCount;

              // Validate bounds before erasing
              if (removePos + removeCount > packetBatch.dispatchPackets.size()) {
                ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                        "[hipGraph] Invalid packet removal bounds: pos=%zu, count=%zu, size=%zu",
                        removePos, removeCount, packetBatch.dispatchPackets.size());
                return hipErrorInvalidValue;
              }

              packetBatch.dispatchPackets.erase(
                  packetBatch.dispatchPackets.begin() + removePos,
                  packetBatch.dispatchPackets.begin() + removePos + removeCount);
              packetBatch.dispatchKernelNames.erase(
                  packetBatch.dispatchKernelNames.begin() + removePos,
                  packetBatch.dispatchKernelNames.begin() + removePos + removeCount);
            }

            // Update this node's packet count and adjust startIndex for all subsequent nodes
            range.packetCount = newPacketCount;
            for (size_t i = rangeIdx + 1; i < packetBatch.nodeRanges.size(); ++i) {
              packetBatch.nodeRanges[i].startIndex = static_cast<size_t>(
                  static_cast<int64_t>(packetBatch.nodeRanges[i].startIndex) + packetDelta);
            }
          }

          // Update dispatch packets (always update regardless of enabled state)
          // The enabled/disabled check happens during dispatch, not here
          for (size_t i = 0; i < range.packetCount && i < newPackets.size(); ++i) {
            size_t packetIndex = range.startIndex + i;
            packetBatch.dispatchPackets[packetIndex] = newPackets[i];
            packetBatch.dispatchKernelNames[packetIndex] = newKernelNames[i];
          }
          return hipSuccess;
        }
      }
      break; // Found the segment, no need to continue
    }
  }
  return hipSuccess; // Node not in any batch
}

// ================================================================================================
hipError_t GraphExec::UpdatePacketBatchesForNodeEnableDisable(hip::GraphNode* node,
                                                              bool isEnabled) {
  if (!node->GraphCaptureEnabled()) {
    // Only handle single stream case with captured nodes
    return hipSuccess;
  }

  // Use node_to_segment_id_ for O(1) segment lookup
  auto segIdIt = node_to_segment_id_.find(node);
  if (segIdIt == node_to_segment_id_.end()) {
    return hipSuccess; // Node not in any segment
  }

  int segmentId = segIdIt->second;

  // Find the segment batch for this segment ID
  for (auto& segBatch : segmentBatches_) {
    if (segBatch.segment_id != segmentId) {
      continue;
    }

    // Search only within this segment's packet batches
    for (auto& packetBatch : segBatch.packet_batches) {
      auto it = packetBatch.nodeToRangeIndex.find(node);
      if (it != packetBatch.nodeToRangeIndex.end()) {
        // Found the batch containing this node - update enabled state
        packetBatch.setEnabled(node, isEnabled);
        return hipSuccess;
      }
    }
    break; // Found the segment, no need to continue
  }
  return hipSuccess;
}

// ================================================================================================

void GraphExec::DecrementRefCount(cl_event event, cl_int command_exec_status, void* user_data) {
  GraphExec* graphExec = reinterpret_cast<GraphExec*>(user_data);
  graphExec->release();
}

// ================================================================================================
hipError_t GraphExec::EnqueueSegmentedGraph(hip::Stream* launch_stream,
                                            const std::vector<hip::Stream*>& parallel_streams) {
  hipError_t status = hipSuccess;

  // Lambda to create and enqueue a marker with wait list
  auto enqueueMarker = [](hip::Stream* stream, const amd::Command::EventWaitList& wait_list) {
    auto marker = new amd::Marker(*stream, true, wait_list);
    marker->setCommandEntryScope(amd::Device::kCacheStateIgnore);
    if (marker != nullptr) {
      marker->enqueue();
      marker->release();
    }
  };

  // Map to track which stream each segment uses
  std::unordered_map<int, hip::Stream*> segment_to_stream;
  // Map to track the last enqueued command on each stream for dependency tracking
  std::unordered_map<hip::Stream*, amd::Command*> stream_last_enqueued_command;

  // Process segments level by level using the pre-calculated max_dependency_level_
  for (int level = 0; level <= max_dependency_level_; ++level) {
    if (segments_per_level_.find(level) == segments_per_level_.end()) {
      continue;
    }

    const auto& segments_at_level = segments_per_level_[level];

    // Assign streams to segments at this level
    for (size_t idx = 0; idx < segments_at_level.size(); ++idx) {
      int segment_id = segments_at_level[idx];
      const auto& segment = segments_[segment_id];

      // Determine device ID for this segment from its first node
      int segment_device_id = launch_stream->DeviceId();
      if (!segment.nodes.empty() && segment.first_node != nullptr) {
        segment_device_id = segment.first_node->GetDeviceId();
      }

      hip::Stream* assigned_stream = nullptr;

      // Use device-aware stream selection with round-robin from parallel_streams_
      if (parallel_streams_.find(segment_device_id) != parallel_streams_.end() &&
          !parallel_streams_[segment_device_id].empty()) {
        // Round-robin across parallel streams for this device + launch stream
        const auto& device_streams = parallel_streams_[segment_device_id];
        size_t stream_idx = idx % (device_streams.size() + 1);
        assigned_stream = (stream_idx == 0) ? launch_stream : device_streams[stream_idx - 1];
      } else {
        // Fallback to launch stream if no parallel streams for this device
        assigned_stream = launch_stream;
      }

      segment_to_stream[segment_id] = assigned_stream;
    }

    // Process each segment at this level
    for (int segment_id : segments_at_level) {
      const auto& segment = segments_[segment_id];
      hip::Stream* current_stream = segment_to_stream[segment_id];

      // Handle dependencies: add wait markers if dependent segments are on different streams
      amd::Command::EventWaitList wait_list;
      for (int dep_segment_id : segment.segment_ids_dependencies) {
        hip::Stream* dep_stream = segment_to_stream[dep_segment_id];

        // Need to wait if dependency is on a different stream
        if (dep_stream != current_stream &&
            stream_last_enqueued_command.find(dep_stream) != stream_last_enqueued_command.end()) {
          amd::Command* dep_command = stream_last_enqueued_command[dep_stream];
          if (dep_command != nullptr) {
            wait_list.push_back(dep_command);
          }
        }
      }

      // If there are cross-stream dependencies, insert a marker to wait
      // Such markers may not flush the caches.
      if (!wait_list.empty()) {
        enqueueMarker(current_stream, wait_list);
      }

      // Create accumulate command for this segment
      amd::AccumulateCommand* accumulate = new amd::AccumulateCommand(*current_stream, {}, nullptr);

      // Enqueue this segment using the helper function
      status = EnqueueSegment(segment, current_stream, accumulate);

      if (status != hipSuccess) {
        accumulate->release();
        // Clean up any previously enqueued commands
        for (auto& pair : stream_last_enqueued_command) {
          if (pair.second != nullptr) {
            pair.second->release();
          }
        }
        return status;
      }

      // Enqueue and track the accumulate command immediately after use
      // Don't release yet - we need to keep it alive for dependencies and final sync
      accumulate->enqueue();

      // If there was a previous accumulate on this stream, release it now
      if (stream_last_enqueued_command.find(current_stream) !=
             stream_last_enqueued_command.end()) {
        amd::Command* prev_cmd = stream_last_enqueued_command[current_stream];
        if (prev_cmd != nullptr) {
          prev_cmd->release();
        }
      }

      stream_last_enqueued_command[current_stream] = accumulate;
    }
  }

  // Synchronize streams from the last dependency level back to launch_stream
  // This ensures launch_stream knows when all work across all devices/streams is complete
  if (max_dependency_level_ >= 0 &&
      segments_per_level_.find(max_dependency_level_) != segments_per_level_.end()) {
    const auto& last_level_segments = segments_per_level_[max_dependency_level_];

    // Collect unique streams used at the last level to sync with launch stream
    // Note: Multiple segments can share the same stream due to round-robin assignment
    // when there are more segments than available parallel streams.
    // We collect unique streams first, then get their last commands to ensure we wait
    // on the most recent work submitted to each stream.
    std::unordered_set<hip::Stream*> last_level_streams;
    for (int segment_id : last_level_segments) {
      hip::Stream* seg_stream = segment_to_stream[segment_id];
      if (seg_stream != launch_stream) {
        last_level_streams.insert(seg_stream);
      }
    }

    // Build wait list from the last enqueued command on each unique stream
    amd::Command::EventWaitList final_wait_list;
    for (hip::Stream* stream : last_level_streams) {
      if (stream_last_enqueued_command.find(stream) != stream_last_enqueued_command.end()) {
        amd::Command* last_cmd = stream_last_enqueued_command[stream];
        if (last_cmd != nullptr) {
          final_wait_list.push_back(last_cmd);
        }
      }
    }

    // If there are other streams at the last level, sync them back to launch_stream
    if (!final_wait_list.empty()) {
      enqueueMarker(launch_stream, final_wait_list);
    }
  }

  // Release all enqueued accumulate commands now that we're done with synchronization
  for (auto& pair : stream_last_enqueued_command) {
    if (pair.second != nullptr) {
      pair.second->release();
    }
  }

  return status;
}

// Enqueue a single segment on a given stream with accumulate command
hipError_t GraphExec::EnqueueSegment(const Segment& segment, hip::Stream* stream,
                                     amd::AccumulateCommand* accumulate) {
  hipError_t status = hipSuccess;

  // Find the SegmentBatch for this segment
  SegmentBatch* segBatch = nullptr;
  for (auto& sb : segmentBatches_) {
    if (sb.segment_id == segment.id) {
      segBatch = &sb;
      break;
    }
  }

  size_t batchIndex = 0;

  // Process all nodes in this segment
  for (size_t i = 0; i < segment.nodes.size(); ++i) {
    auto& node = segment.nodes[i];

    if (!node->GraphCaptureEnabled()) {
      // Node doesn't support capture - execute individually
      node->SetStream(stream);
      status = node->CreateCommand(node->GetQueue());
      node->EnqueueCommands(stream);
    } else if (segBatch && i < segBatch->node_capture_status.size() &&
               segBatch->node_capture_status[i]) {
      // Node was successfully captured - dispatch its batch
      if (segBatch && batchIndex < segBatch->packet_batches.size()) {
        auto& packetBatch = segBatch->packet_batches[batchIndex];

        // Select which vectors to dispatch based on whether nodes are disabled
        const std::vector<uint8_t*>* packetsToDispatch;
        const std::vector<std::string>* kernelNamesToDispatch;

        if (packetBatch.disabledNodeCount == 0) {
          // No disabled nodes - use full batch
          packetsToDispatch = &packetBatch.dispatchPackets;
          kernelNamesToDispatch = &packetBatch.dispatchKernelNames;
        } else {
          // Some nodes disabled - rebuild and use filtered lists
          packetBatch.rebuildFilteredLists();
          packetsToDispatch = &packetBatch.enabledPackets;
          kernelNamesToDispatch = &packetBatch.enabledKernelNames;
        }

        // Dispatch the selected batch
        if (!packetsToDispatch->empty()) {
          bool batchStatus = stream->vdev()->dispatchAqlPacketBatch(
              *packetsToDispatch, *kernelNamesToDispatch, accumulate);
          if (!batchStatus) {
            status = hipErrorUnknown;
            return status;
          }
        }

        // Skip all consecutive captured nodes that belong to this batch
        i += packetBatch.nodeRanges.size() - 1;  // -1 because loop will increment
        ++batchIndex;
      }
    }
  }

  return status;
}
// ================================================================================================
void GraphExec::UpdateStreams(hip::Stream* launch_stream) {
  int devId = launch_stream->vdev()->device().index();
  // Current stream is the default in the assignment
  streams_.push_back(launch_stream);
  if (parallel_streams_.find(devId) == parallel_streams_.end()) {
    LogPrintfError("UpdateStreams failed for device id:%d", devId);
    return;
  }
  auto parallel_streams = parallel_streams_[devId];
  std::unordered_map<int, int> unique_stream_ids;
  unique_stream_ids[launch_stream->getQueueID()] = 1;
  std::vector<hip::Stream*> collided_streams;
  // Assign streams that are unique in parallel_streams and doesnt collide with launch stream
  for (uint32_t i = 0; i < parallel_streams.size(); i++) {
    auto qid = parallel_streams[i]->getQueueID();
    if (unique_stream_ids[qid] == 0) {
      streams_.push_back(parallel_streams[i]);
    } else {
      collided_streams.push_back(parallel_streams[i]);
    }
    unique_stream_ids[qid]++;
  }
  // Assign the remaining streams for execution.
  for (int i = streams_.size(), j = 0; i < max_streams_ && j < collided_streams.size(); i++, j++) {
    streams_.push_back(collided_streams[j]);
  }
}


// ================================================================================================
bool Graph::RunOneNode(Node node, bool wait) {
  if (node->launch_id_ == -1) {
    // Clear the storage of the wait nodes
    memset(&wait_order_[0], 0, sizeof(Node) * wait_order_.size());
    amd::Command::EventWaitList waitList;
    // Walk through dependencies and find the last launches on each parallel stream
    for (auto depNode : node->GetDependencies()) {
      // Process only the nodes that have been submitted
      if (depNode->launch_id_ != -1) {
        // If it's the same stream then skip the signal, since it's in order
        if (depNode->stream_id_ != node->stream_id_) {
          // If there is no wait node on the stream, then assign one
          if ((wait_order_[depNode->stream_id_] == nullptr) ||
              // If another node executed on the same stream, then use the latest launch only,
              // since the same stream has in-order run
              (wait_order_[depNode->stream_id_]->launch_id_ < depNode->launch_id_)) {
            wait_order_[depNode->stream_id_] = depNode;
          }
        }
      } else {
        // It should be a safe return,
        // since the last edge to this dependency has to submit the command
        return true;
      }
    }

    // Create a wait list from the last launches of all dependencies
    for (auto dep : wait_order_) {
      if (dep != nullptr) {
        // Add all commands in the wait list
        if (dep->GetType() != hipGraphNodeTypeGraph) {
          for (auto command : dep->GetCommands()) {
            waitList.push_back(command);
          }
        }
      }
    }
    if (node->GetType() == hipGraphNodeTypeGraph) {
      // Process child graph separately, since, there is no connection
      auto child = reinterpret_cast<hip::ChildGraphNode*>(node)->GetChildGraph();
      if (!reinterpret_cast<hip::ChildGraphNode*>(node)->GetGraphCaptureStatus()) {
        child->RunNodes(node->stream_id_, &streams_, &waitList);
      }
    } else {
      // Assing a stream to the current node
      node->SetStream(streams_);
      // Create the execution commands on the assigned stream
      auto status = node->CreateCommand(node->GetQueue());
      if (status != hipSuccess) {
        LogPrintfError("Command creation for node id(%d) failed!", current_id_ + 1);
        return false;
      }
      // Retain all commands, since potentially the command can finish before a wait signal
      for (auto command : node->GetCommands()) {
        command->retain();
      }

      // If a wait was requested, then process the list
      if (wait && !waitList.empty()) {
        node->UpdateEventWaitLists(waitList);
      }
      // Start the execution
      node->EnqueueCommands(node->GetQueue());
    }
    // Assign the launch ID of the submmitted node
    // This is also applied to childGraphs to prevent them from being reprocessed
    node->launch_id_ = current_id_++;
    uint32_t i = 0;
    // Execute the nodes in the edges list
    for (auto edge : node->GetEdges()) {
      // Don't wait in the nodes, executed on the same streams and if it has just one dependency
      bool wait = ((i < DEBUG_HIP_FORCE_GRAPH_QUEUES) || (edge->GetDependencies().size() > 1))
                      ? true
                      : false;
      // Execute the edge node
      if (!RunOneNode(edge, wait)) {
        return false;
      }
      i++;
    }
    if (i == 0) {
      // Add a leaf node into the list for a wait.
      // Always use the last node, since it's the latest for the particular queue
      leafs_[node->stream_id_] = node;
    }
  }
  return true;
}

// ================================================================================================
bool Graph::RunNodes(int32_t base_stream, const std::vector<hip::Stream*>* parallel_streams,
                     const amd::Command::EventWaitList* parent_waitlist) {
  if (parallel_streams != nullptr) {
    streams_ = *parallel_streams;
  }

  // childgraph node has dependencies on parent graph nodes from other streams
  if (parent_waitlist != nullptr) {
    auto start_marker = new amd::Marker(*streams_[base_stream], true, *parent_waitlist);
    if (start_marker != nullptr) {
      start_marker->enqueue();
      start_marker->release();
    }
  }
  amd::Command::EventWaitList wait_list;
  current_id_ = 0;
  memset(&leafs_[0], 0, sizeof(Node) * leafs_.size());

  // Add possible waits in parallel streams for the app's default launch stream
  constexpr bool kRetainCommand = true;
  auto last_command = streams_[base_stream]->getLastQueuedCommand(kRetainCommand);
  if (last_command != nullptr) {
    // Add the last command into the waiting list
    wait_list.push_back(last_command);
    // Check if the graph has multiple root nodes
    for (uint32_t i = 0; i < DEBUG_HIP_FORCE_GRAPH_QUEUES; ++i) {
      if ((base_stream != i) && (roots_[i] != nullptr)) {
        // Wait for the app's queue
        auto start_marker = new amd::Marker(*streams_[i], true, wait_list);
        if (start_marker != nullptr) {
          start_marker->enqueue();
          start_marker->release();
        }
      }
    }
    last_command->release();
  }

  // Run all commands in the graph
  for (auto node : vertices_) {
    if (node->launch_id_ == -1) {
      if (!RunOneNode(node, true)) {
        return false;
      }
    }
  }
  wait_list.clear();
  // Check if the graph has multiple leaf nodes
  for (uint32_t i = 0; i < DEBUG_HIP_FORCE_GRAPH_QUEUES; ++i) {
    if ((base_stream != i) && (leafs_[i] != nullptr)) {
      // Add all commands in the wait list
      if (leafs_[i]->GetType() != hipGraphNodeTypeGraph) {
        for (auto command : leafs_[i]->GetCommands()) {
          wait_list.push_back(command);
        }
      }
    }
  }
  // Wait for leafs in the graph's app stream
  if (wait_list.size() > 0) {
    auto end_marker = new amd::Marker(*streams_[base_stream], true, wait_list);
    if (end_marker != nullptr) {
      end_marker->enqueue();
      end_marker->release();
    }
  }
  // Release commands after execution
  for (auto& node : vertices_) {
    node->launch_id_ = -1;
    if (node->GetType() != hipGraphNodeTypeGraph) {
      for (auto command : node->GetCommands()) {
        command->release();
      }
    }
  }
  return true;
}

// ================================================================================================
hipError_t GraphExec::Run(hip::Stream* launch_stream) {
  hipError_t status = hipSuccess;

  // Get the first node based on scheduling mode
  Node firstNode = nullptr;
  if (DEBUG_HIP_GRAPH_PACKET_ENGINE && !segments_.empty() && !segments_[0].nodes.empty()) {
    firstNode = segments_[0].nodes[0];
  } else if (!topoOrder_.empty()) {
    firstNode = topoOrder_[0];
  }

  if (flags_ & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    if (firstNode != nullptr) {
      firstNode->GetParentGraph()->FreeAllMemory(launch_stream);
      firstNode->GetParentGraph()->memalloc_nodes_ = 0;
      if (!AMD_DIRECT_DISPATCH) {
        // The MemoryPool::FreeAllMemory queues a memory unmap command that for !AMD_DIRECT_DISPATCH
        // runs asynchonously. Make sure that freeAllMemory is complete before creating new commands
        // to prevent races to the MemObjMap.
        launch_stream->finish();
      }
    }
  }

  // If this is a repeat launch, make sure corresponding MemFreeNode exists for a MemAlloc node
  if (repeatLaunch_ == true) {
    if (firstNode != nullptr && firstNode->GetParentGraph()->GetMemAllocNodeCount() > 0) {
      return hipErrorInvalidValue;
    }
  } else {
    repeatLaunch_ = true;
  }

  ClPrint(amd::LOG_DEBUG, amd::LOG_CODE, "GraphExec::Run max_streams: %d, on device: %d",
          max_streams_, launch_stream->DeviceId());

  if (DEBUG_HIP_GRAPH_PACKET_ENGINE && instantiateDeviceId_ == launch_stream->DeviceId()) {
    // If the graph has kernels that does device side allocation,  during packet capture, heap is
    // allocated because heap pointer has to be added to the AQL packet, and initialized during
    // graph launch.
    static bool initialized = false;
    // Todo: Hidden heap initialization is done only for single device graph
    if (!initialized && HasHiddenHeap()) {
      launch_stream->vdev()->HiddenHeapInit();
      initialized = true;
    }
    if (max_streams_dev_.size() == 1) {
      status = EnqueueSegmentedGraph(launch_stream, parallel_streams_[launch_stream->DeviceId()]);
    } else {
      // Multi-device graph: EnqueueSegmentedGraph now handles multi-device synchronization
      // Pass empty parallel_streams since it uses parallel_streams_ member variable internally
      status = EnqueueSegmentedGraph(launch_stream, {});
    }
  } else if (max_streams_ == 1 && instantiateDeviceId_ != launch_stream->DeviceId()) {
    for (int i = 0; i < topoOrder_.size(); i++) {
      topoOrder_[i]->SetStream(launch_stream);
      status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
      topoOrder_[i]->EnqueueCommands(launch_stream);
    }
  } else {
    // Update streams for the graph execution
    UpdateStreams(launch_stream);
    // Execute all nodes in the graph
    if (!RunNodes()) {
      LogError("Failed to launch nodes!");
      return hipErrorOutOfMemory;
    }
  }
  this->retain();
  amd::Command* CallbackCommand = new amd::Marker(*launch_stream, kMarkerDisableFlush, {});
  // we may not need to flush any caches.
  CallbackCommand->setCommandEntryScope(amd::Device::kCacheStateIgnore);
  amd::Event& event = CallbackCommand->event();
  constexpr bool kBlocking = false;
  if (!event.setCallback(CL_COMPLETE, GraphExec::DecrementRefCount, this, kBlocking)) {
    return hipErrorInvalidHandle;
  }
  CallbackCommand->enqueue();
  CallbackCommand->release();
  return status;
}

// ================================================================================================
bool GraphKernelArgManager::AllocGraphKernargPool(size_t pool_size, amd::Device* device) {
  bool bStatus = true;
  assert(pool_size > 0);
  address graph_kernarg_base;
  if (device->info().largeBar_) {
    amd::Device::AllocationFlags flags = {};
    flags.executable_ = true;
    graph_kernarg_base = reinterpret_cast<address>(device->deviceLocalAlloc(pool_size, flags));
    device_kernarg_pool_ = true;
  } else {
    graph_kernarg_base = reinterpret_cast<address>(
        device->hostAlloc(pool_size, 0, amd::Device::MemorySegment::kKernArg));
  }

  if (graph_kernarg_base == nullptr) {
    return false;
  }
  kernarg_graph_[device].push_back(KernelArgPoolGraph(graph_kernarg_base, pool_size));
  return true;
}

address GraphKernelArgManager::AllocKernArg(size_t size, size_t alignment, int devId) {
  if (size == 0) {
    return nullptr;
  }

  amd::Device* device = g_devices[devId]->devices()[0];
  assert(alignment != 0 && "Alignment must be non-zero");

  // Check if we have any pools allocated for this device
  auto& device_pools = kernarg_graph_[device];
  if (device_pools.empty()) {
    return nullptr;
  }

  auto& current_pool = device_pools.back();
  // Calculate aligned address for the allocation
  address aligned_addr = amd::alignUp(current_pool.kernarg_pool_addr_ + current_pool.kernarg_pool_offset_, alignment);
  const size_t new_pool_usage = (aligned_addr + size) - current_pool.kernarg_pool_addr_;

  // Check if allocation fits in current pool
  if (new_pool_usage <= current_pool.kernarg_pool_size_) {
    current_pool.kernarg_pool_offset_ = new_pool_usage;
    return aligned_addr;
  }

  // Current pool is full - allocate a new pool with the same size
  if (!AllocGraphKernargPool(current_pool.kernarg_pool_size_, device)) {
    return nullptr;
  }

  // Recursively allocate from the new pool
  return AllocKernArg(size, alignment, devId);
}

void GraphKernelArgManager::ReadBackOrFlush() {
  if (!device_kernarg_pool_) {
    return;
  }

  for (const auto& kernarg : kernarg_graph_) {
    const auto kernArgImpl = kernarg.first->settings().kernel_arg_impl_;

    if (kernArgImpl == KernelArgImpl::DeviceKernelArgsHDP) {
      // Trigger HDP flush
      *kernarg.first->info().hdpMemFlushCntl = 1u;
      // Read back to ensure flush completion
      volatile int kSentinel = *reinterpret_cast<volatile int*>(kernarg.first->info().hdpMemFlushCntl);
      (void)kSentinel; // Suppress unused variable warning
    } else if (kernArgImpl == KernelArgImpl::DeviceKernelArgsReadback) {
      const auto& pool = kernarg.second.back();
      if (pool.kernarg_pool_addr_ == 0) {
        continue;
      }

      // Perform readback operation on the last byte of the pool
      address dev_ptr = pool.kernarg_pool_addr_ + pool.kernarg_pool_size_;
      volatile unsigned char* sentinel_ptr = reinterpret_cast<volatile unsigned char*>(dev_ptr - 1);

      // Read-modify-write sequence with memory barriers
      volatile unsigned char kSentinel = *sentinel_ptr;
      _mm_sfence();
      *sentinel_ptr = kSentinel;
      _mm_mfence();
      kSentinel = *sentinel_ptr;
      (void)kSentinel; // Suppress unused variable warning
    }
  }
}
}  // namespace hip
