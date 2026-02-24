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
  return roots;
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
      reinterpret_cast<hip::ChildGraphNode*>(node)->GraphExec::TopologicalOrder();
    }
    for (auto edge : node->GetEdges()) {
      if (edge->stream_id_ == -1) {
        ScheduleOneNode(edge, stream_id);
        // 1. Each extra edge will get a new stream from the pool
        // 2. Streams will be reused if the number of edges > streams
        stream_id = (stream_id + 1) % DEBUG_HIP_FORCE_GRAPH_QUEUES;
      }
    }
  }
}

// ================================================================================================
hipError_t Graph::ScheduleNodes() {
  if (use_segment_scheduling_) {
    // Segment packet scheduling logic
    hipError_t result = ScheduleNodesIntoBatches();

    // If ScheduleNodesIntoBatches returns hipErrorNotReady, it indicates
    // a complex graph that would benefit from classic path, so fall back
    if (result == hipErrorNotReady) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
              "[hipGraph] Falling back to classic scheduling for complex graph");
      // Clear any partial segment data that might have been created
      segments_.clear();
      node_to_segment_id_.clear();
      segments_per_level_.clear();
      max_dependency_level_ = -1;
      // Disable segment scheduling for this graph permanently
      use_segment_scheduling_ = false;

      // Continue to classic scheduling logic below
    } else {
      // Return success or actual error (not the special fallback indicator)
      return result;
    }
  }

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
  GraphExec* graphExec = dynamic_cast<GraphExec*>(this);
  if (graphExec && !graphExec->TopologicalOrder()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] TopologicalOrder failed - invalid graph");
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t Graph::ScheduleNodesIntoBatches() {
  // Handle empty graph case - valid, nothing to schedule
  if (GetNodeCount() == 0) {
    return hipSuccess;
  }

  // Find execution paths hierarchically (new approach)
  auto hierarchical_paths = FindExecutionPathsHierarchical();
  if (hierarchical_paths.paths.empty()) {
    // If we have nodes but no paths, this indicates an invalid graph (likely a cycle)
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] No execution paths found - graph may contain cycles");
    return hipErrorInvalidValue;
  }

  // Create segments from hierarchical paths (new approach)
  CreateSegmentsFromPaths(hierarchical_paths);
  // Verify we created at least one valid segment
  if (segments_.empty()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] No valid segments created from execution paths");
    return hipErrorInvalidValue;
  }

  // Check if this is a complex graph that would benefit from classic path
  // Complex graphs: 16+ segments with average segment length < 8
  const size_t kSegmentSizeThreshold = 16;
  const double kAvgSegmentLengthThreshold = 8.0;
  if (segments_.size() >= kSegmentSizeThreshold && DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING != 2) {
    size_t total_nodes = 0;
    for (const auto& segment : segments_) {
      total_nodes += segment.nodes.size();
    }
    double avg_segment_length = static_cast<double>(total_nodes) / segments_.size();

    if (avg_segment_length < kAvgSegmentLengthThreshold) {
      ClPrint(amd::LOG_INFO, amd::LOG_CODE,
              "[hipGraph] Complex graph detected: %zu segments, avg length %.2f - "
              "falling back to classic path for better performance",
              segments_.size(), avg_segment_length);
      // Return special status to indicate fallback to classic path
      return hipErrorNotReady;
    }
  }

  // Resolve segment dependencies and calculate dependency levels
  ResolveSegmentDependencies();

  // Calculate topological order for fallback paths and compatibility
  // (e.g., child graphs, legacy execution, GetNodes() API)
  GraphExec* graphExec = dynamic_cast<GraphExec*>(this);
  if (graphExec && !graphExec->TopologicalOrder()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] TopologicalOrder failed - graph may contain cycles");
    return hipErrorInvalidValue;
  }

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
          "[hipGraph] ScheduleNodesIntoBatches: Total nodes = %zu, total segments = %zu max "
          "dependency level = %d, max streams = %d",
          GetNodeCount(), segments_.size(), max_dependency_level_, max_streams_);

  return hipSuccess;
}

// ================================================================================================
void Graph::ResolveSegmentDependencies() {
  // Resolve dependencies within this graph
  for (size_t i = 0; i < segments_.size(); ++i) {
    auto& segment = segments_[i];

    // Only check first node for incoming dependencies
    if (segment.first_node != nullptr) {
      const auto& dependencies = segment.first_node->GetDependencies();

      for (const auto& dep_node : dependencies) {
        // Find which segment this dependency belongs to (within this graph)
        auto dep_it = node_to_segment_id_.find(dep_node);
        if (dep_it != node_to_segment_id_.end()) {
          int dep_segment_id = dep_it->second;

          // Validate segment ID is within bounds
          if (dep_segment_id < 0 || dep_segment_id >= static_cast<int>(segments_.size())) {
            ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                    "[hipGraph] Invalid segment ID %d (segments size: %zu)",
                    dep_segment_id, segments_.size());
            continue;  // Skip invalid segment ID
          }

          // Add dependency if not already present
          if (std::find(segment.segment_ids_dependencies.begin(),
                       segment.segment_ids_dependencies.end(),
                       dep_segment_id) == segment.segment_ids_dependencies.end()) {
            segment.segment_ids_dependencies.push_back(dep_segment_id);

            // Also add this segment as an edge of the dependency segment
            segments_[dep_segment_id].segment_ids_edges.push_back(i);
          }
        }
      }
    }
  }

  // Recursively resolve dependencies in child graphs
  // When a parent segment depends on a segment containing a child graph node,
  // it implicitly depends on ALL segments in that child graph completing.
  for (auto& segment : segments_) {
    if (segment.child_graph_ptr != nullptr) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
              "[hipGraph] Recursively resolving dependencies"
              "for child graph %p in segment [id=%d]",
              segment.child_graph_ptr, segment.id);

      // Child graph resolves its own internal segment dependencies
      segment.child_graph_ptr->ResolveSegmentDependencies();
    }
  }

  // Calculate dependency levels and max_streams_ using topological sort
  CalculateSegmentTopoDependencyLevels();
}

// ================================================================================================
void Graph::CalculateSegmentTopoDependencyLevels() {
  // Topological sort of segments to calculate dependency levels
  // Assume each segment is a node and the dependencies are segments edges
  // Segments with same dependency level can be processed in parallel
  std::queue<int> queue;
  std::unordered_map<int, int> in_degree;

  // Reset max dependency level, max streams, and segments per level
  max_dependency_level_ = -1;
  max_streams_ = 1;
  segments_per_level_.clear();

  // Initialize in-degree for each segment and enqueue root segments
  for (size_t i = 0; i < segments_.size(); ++i) {
    segments_[i].dependency_level = -1;
    in_degree[i] = segments_[i].segment_ids_dependencies.size();

    if (in_degree[i] == 0) {
      // Root segments have level 0
      segments_[i].dependency_level = 0;
      queue.push(i);
      max_dependency_level_ = 0;
      segments_per_level_[0].push_back(i);
    }
  }

  // Process segments in topological order
  while (!queue.empty()) {
    int current_id = queue.front();
    queue.pop();

    auto& current_segment = segments_[current_id];
    int current_level = current_segment.dependency_level;

    // Process all segments that depend on current segment
    for (int edge_id : current_segment.segment_ids_edges) {
      auto& edge_segment = segments_[edge_id];

      // Calculate the dependency level for this segment
      // It's one level higher than the maximum of its dependencies
      int new_level = current_level + 1;
      if (edge_segment.dependency_level < new_level) {
        edge_segment.dependency_level = new_level;
        // Track the maximum dependency level
        max_dependency_level_ = std::max(max_dependency_level_, new_level);
      }

      // Decrease in-degree and enqueue if all dependencies processed
      in_degree[edge_id]--;
      if (in_degree[edge_id] == 0) {
        queue.push(edge_id);
        // Add segment to its dependency level
        segments_per_level_[edge_segment.dependency_level].push_back(edge_id);
      }
    }
  }

  // Calculate max_streams_ based on maximum parallelism at any dependency level
  for (const auto& level_segments : segments_per_level_) {
    max_streams_ = std::max(max_streams_, static_cast<int>(level_segments.second.size()));
  }
}

// ================================================================================================
hip::Graph::GraphExecutionPaths Graph::FindExecutionPathsHierarchical() {
  hip::Graph::GraphExecutionPaths graph_paths;
  graph_paths.graph_ptr = this;

  // Find all root nodes (nodes with no dependencies)
  const auto& root_nodes = GetRootNodes();

  std::unordered_set<unsigned int> visited;
  for (const auto& root : root_nodes) {
    // For each root, find all possible paths starting from it
    std::vector<Node> current_path;
    FindPathsRecursiveHierarchical(root, current_path, visited, graph_paths);
  }
  return graph_paths;
}

// ================================================================================================
void Graph::FindPathsRecursiveHierarchical(Node node,
                                           std::vector<Node>& current_path,
                                           std::unordered_set<unsigned int>& visited,
                                           hip::Graph::GraphExecutionPaths& graph_paths) {
  // Lambda to save current path as a HierarchicalPath
  auto savePath = [&graph_paths](const std::vector<Node>& path, int device_id,
                                  Node child_node = nullptr, int child_index = -1) {
    hip::Graph::HierarchicalPath h_path;
    h_path.nodes = path;
    h_path.device_id = device_id;
    h_path.child_graph_node = child_node;
    h_path.child_graph_paths_index = child_index;
    graph_paths.paths.push_back(std::move(h_path));
  };

  // Check if already visited
  if (visited.find(node->GetID()) != visited.end()) {
    return;
  }

  // Mark regular nodes as visited
  visited.insert(node->GetID());

  // Check if device ID changed from previous node in path
  bool device_changed = false;
  int current_device_id = node->GetDeviceId();
  if (!current_path.empty()) {
    int prev_device_id = current_path.back()->GetDeviceId();
    if (prev_device_id != current_device_id) {
      device_changed = true;
      // Save current path before device change
      savePath(current_path, prev_device_id);
      current_path.clear();
    }
  }

  // Handle child graph nodes specially
  if (node->GetType() == hipGraphNodeTypeGraph) {
    // Save path before child graph node (if any)
    if (!current_path.empty()) {
      savePath(current_path, current_path.back()->GetDeviceId());
      current_path.clear();
    }

    // Get the child graph and recursively process it
    auto childGraphNode = reinterpret_cast<hip::ChildGraphNode*>(node);
    auto childGraph = childGraphNode->GetChildGraph();

    if (childGraph != nullptr) {
      // Create a new GraphExecutionPaths for this child graph
      hip::Graph::GraphExecutionPaths child_graph_exec_paths;
      child_graph_exec_paths.graph_ptr = childGraph;

      // Find all root nodes in the child graph
      const auto& child_root_nodes = childGraph->GetRootNodes();
      std::unordered_set<unsigned int> child_visited;

      for (const auto& child_root : child_root_nodes) {
        std::vector<Node> child_current_path;
        childGraph->FindPathsRecursiveHierarchical(child_root, child_current_path,
                                                   child_visited, child_graph_exec_paths);
      }

      // Store the child graph paths
      int child_graph_index = graph_paths.child_graph_paths.size();
      graph_paths.child_graph_paths.push_back(std::move(child_graph_exec_paths));

      // Create a path containing just the child graph node
      std::vector<Node> child_node_path = {childGraphNode};
      savePath(child_node_path, current_device_id, childGraphNode, child_graph_index);
    }

    // Clear current path and continue with edges from the child graph node
    current_path.clear();
    const auto& edges = node->GetEdges();
    for (const auto& edge : edges) {
      FindPathsRecursiveHierarchical(edge, current_path, visited, graph_paths);
    }

    return;
  }

  // Regular node - add to current path
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

      if (!path_to_save.empty()) {
        savePath(path_to_save, path_to_save.back()->GetDeviceId());
      }
      current_path.clear();

      // For nodes that are both fork and join, save them as their own segment
      if (saved_join_node != nullptr && is_fork) {
        std::vector<Node> fork_join_segment = {saved_join_node};
        savePath(fork_join_segment, saved_join_node->GetDeviceId());
      }

      // Put the join node back in current_path for further traversal
      // But not if it's also a fork node, because we'll traverse branches separately
      if (saved_join_node != nullptr && !is_fork) {
        current_path.push_back(saved_join_node);
      }
    }

    // Traverse each branch until it hits a join
    for (const auto& edge : edges) {
      FindPathsRecursiveHierarchical(edge, current_path, visited, graph_paths);

      // Save the path if it's not empty and this was a fork/join boundary
      if (!current_path.empty() && (is_fork || is_join)) {
        savePath(current_path, current_path.back()->GetDeviceId());
        current_path.clear();
      }
    }
  } else if (edges.size() == 1) {
    // Single edge - continue on same path
    FindPathsRecursiveHierarchical(edges[0], current_path, visited, graph_paths);
  }

  // Save any remaining path (handles leaf nodes and leaf join nodes)
  if (!current_path.empty()) {
    savePath(current_path, current_path.back()->GetDeviceId());
    current_path.clear();
  }
}

// ================================================================================================
void Graph::CreateSegmentsFromPaths(const hip::Graph::GraphExecutionPaths& exec_paths) {
  // Clear previous segments
  segments_.clear();
  node_to_segment_id_.clear();

  // Create a segment for each execution path at this level
  int segment_id = 0;
  for (size_t i = 0; i < exec_paths.paths.size(); ++i) {
    const auto& h_path = exec_paths.paths[i];
    if (h_path.nodes.empty()) continue;

    Segment segment;
    segment.id = segment_id;
    segment.nodes = h_path.nodes;
    segment.first_node = h_path.nodes.front();
    segment.last_node = h_path.nodes.back();

    // Preserve child graph information from hierarchical path
    if (h_path.child_graph_node != nullptr && h_path.child_graph_paths_index >= 0) {
      // Get direct pointer to child graph from the node
      auto childGraphNode = reinterpret_cast<hip::ChildGraphNode*>(h_path.child_graph_node);
      segment.child_graph_ptr = childGraphNode->GetChildGraph();
    }

    segments_.push_back(segment);

    // Map each node in this segment to the segment ID (local to this graph)
    for (const auto& node : segment.nodes) {
      node_to_segment_id_[node] = segment_id;
      node->segment_id_ = segment_id;
    }

    segment_id++;
  }

  // Recursively process child graphs
  for (size_t i = 0; i < exec_paths.child_graph_paths.size(); ++i) {
    const auto& child_paths = exec_paths.child_graph_paths[i];

    if (child_paths.graph_ptr != nullptr) {
      // Let the child graph create its own segments
      child_paths.graph_ptr->CreateSegmentsFromPaths(child_paths);
    }
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
  // Map original root node pointers to their cloned counterparts
  if (roots_.size() > 0) {
    for (size_t i = 0; i < roots_.size(); ++i) {
      if (roots_[i] != nullptr) {
        auto it = newGraph->clonedNodes_.find(roots_[i]);
        if (it != newGraph->clonedNodes_.end()) {
          newGraph->roots_[i] = it->second;
        } else {
          newGraph->roots_[i] = nullptr;
        }
      } else {
        newGraph->roots_[i] = nullptr;
      }
    }
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
    auto stream = new hip::Stream(g_devices[devId], hip::Stream::Priority::Normal,
                                  hipStreamNonBlocking);

    if (!stream->Create()) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to create stream %u for device %d",
              i, devId);
      hip::Stream::Destroy(stream);
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

  // Account for the launch stream that's available only on the instantiation device
  // We only need to create (count - 1) extra streams for the instantiation device
  for (auto& [dev_id, count] : max_streams_dev_) {
    if (dev_id == instantiateDeviceId_ && count > 0) {
      count = count - 1;
    }
  }
}

// ================================================================================================
void GraphExec::FindStreamsReqPerDevForSegments() {
  // For packet engine mode: analyze segments to determine stream requirements per device
  // We need to track the maximum number of concurrent segments per device at any level

  std::unordered_map<int, int> streams_per_dev_at_level;

  for (const auto& [level, segment_ids] : segments_per_level_) {
    streams_per_dev_at_level.clear();

    // Count segments per device at this level
    for (int segment_id : segment_ids) {
      if (segment_id >= 0 && segment_id < static_cast<int>(segments_.size())) {
        const auto& segment = segments_[segment_id];

        // Determine device ID from segment's first node
        int dev_id = hip::getCurrentDevice()->deviceId();
        if (!segment.nodes.empty() && segment.first_node != nullptr) {
          dev_id = segment.first_node->GetDeviceId();
        }

        streams_per_dev_at_level[dev_id]++;
      }
    }

    // Update max streams per device based on this level's requirements
    for (const auto& [dev_id, count] : streams_per_dev_at_level) {
      max_streams_dev_[dev_id] = std::max(max_streams_dev_[dev_id], count);
    }
  }

  // Account for the launch stream that's available only on the instantiation device
  // We only need to create (count - 1) extra streams for the instantiation device
  for (auto& [dev_id, count] : max_streams_dev_) {
    if (dev_id == instantiateDeviceId_ && count > 0) {
      count = count - 1;
    }
  }
}

// ================================================================================================
hipError_t GraphExec::Init() {
  hipError_t status = hipSuccess;
  // Set instantiation device ID early so Find functions can use it
  instantiateDeviceId_ = hip::getCurrentDevice()->deviceId();

  // create extra stream to avoid queue collision with the default execution stream
  if (max_streams_ >= 1) {
    if (use_segment_scheduling_) {
      // For packet engine: analyze segments to determine per-device stream requirements
      FindStreamsReqPerDevForSegments();
    } else {
      // For classic scheduling: use stream-to-device mappings
      FindStreamsReqPerDev();
    }

    // Create parallel streams for each device based on computed requirements
    // Note: max_streams_dev_ already accounts for the launch stream, so it contains
    // the number of extra streams to create
    for (auto const& [dev_id, num_streams] : max_streams_dev_) {
      if (num_streams > 0) {
        status = CreateStreams(num_streams, dev_id);
        if (status != hipSuccess) {
          return status;
        }
      } else {
        // No extra streams needed
      }
    }
  }

  if (use_segment_scheduling_) {
    // For graph nodes capture AQL packets to dispatch them directly during graph launch.
    status = CaptureAQLPackets();
  }

  static_cast<ReferenceCountedObject*>(hip::getCurrentDevice())->retain();
  return status;
}

//! Chunk size to add to kern arg pool
constexpr uint32_t kKernArgChunkSize = 128 * Ki;
// ================================================================================================
void GraphExec::GetKernelArgSizeForGraph(std::unordered_map<int, size_t>& kernArgSizeForGraph) {
  // Calculate the kernel argument size required for all graph kernel nodes
  // when GPU packet capture is enabled

  if (use_segment_scheduling_ && !segments_.empty()) {
    for (const auto& segment : segments_) {
      // Handle child graph segments - skip node iteration, process recursively
      if (segment.child_graph_ptr != nullptr) {
        auto childGraphExec = dynamic_cast<GraphExec*>(segment.child_graph_ptr);
        if (childGraphExec != nullptr) {
          // Child graphs share the same kernel arg manager as parent
          if (childGraphExec->GetKernelArgManager() == nullptr) {
            auto kernArgMgr = GetKernelArgManager();
            if (kernArgMgr != nullptr) {
              kernArgMgr->retain();  // Increment ref count for child's reference
              childGraphExec->SetKernelArgManager(kernArgMgr);
            }
          }
          childGraphExec->GetKernelArgSizeForGraph(kernArgSizeForGraph);
        }
        continue;  // Skip processing nodes in this segment
      }

      // Process regular nodes in this segment
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
    // Defensive check to prevent underflow
    if (disabledNodeCount > 0) {
      disabledNodeCount--;
    }
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
    // Skip segments that only contain a child graph metadata node
    // Child graphs are processed recursively later
    if (segment.child_graph_ptr != nullptr) {
      continue;
    }

    // Create a SegmentBatch for this segment
    auto [it, inserted] = segmentBatches_.emplace(segment.id, segment.id);
    // Initialize node_capture_status for this segment
    auto& currentSegBatch = it->second;
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
      }
    }
  }

  // Recursively process child graphs to capture their packets
  for (const auto& segment : segments_) {
    if (segment.child_graph_ptr != nullptr) {
      auto childGraphExec = dynamic_cast<GraphExec*>(segment.child_graph_ptr);
      if (childGraphExec != nullptr) {
        // Child graphs share the same kernel arg manager as parent
        // This is critical for packet capture to work correctly
        if (childGraphExec->GetKernelArgManager() == nullptr) {
          auto kernArgMgr = GetKernelArgManager();
          if (kernArgMgr != nullptr) {
            kernArgMgr->retain();  // Increment ref count for child's reference
            childGraphExec->SetKernelArgManager(kernArgMgr);
          }
        }

        status = childGraphExec->CaptureAndFormPacketsForGraph();
        if (status != hipSuccess) {
          LogWarning("Child graph packet capture failed for child graph in segment");
          // Continue processing other child graphs
          status = hipSuccess;
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
  // Use node_to_segment_id_ for O(1) segment lookup
  auto segIdIt = node_to_segment_id_.find(node);
  if (segIdIt == node_to_segment_id_.end()) {
    return hipSuccess;  // Node not in any segment
  }

  int segmentId = segIdIt->second;

  // Find the segment batch for this segment ID using O(1) map lookup
  auto segBatchIt = segmentBatches_.find(segmentId);
  if (segBatchIt == segmentBatches_.end()) {
    return hipSuccess;  // Segment not found
  }

  auto& segBatch = segBatchIt->second;

  // Search only within this segment's packet batches
  for (auto& packetBatch : segBatch.packet_batches) {
    auto it = packetBatch.nodeToRangeIndex.find(node);
    if (it != packetBatch.nodeToRangeIndex.end()) {
      // Found the batch containing this node - update packets
      PacketBatch::NodeRange& range = packetBatch.nodeRanges[it->second];

      // Capture new packets for this node
      std::vector<uint8_t*> newPackets;
      std::vector<std::string> newKernelNames;
      hipError_t status = node->CaptureAndFormPacket(kernArgManager_, &newPackets,
                                                                      &newKernelNames);
      if (status != hipSuccess) {
        return status;
      }
      // Number of packets per node can change
      const size_t oldPacketCount = range.packetCount;
      const size_t newPacketCount = newPackets.size();

      if (newPacketCount != oldPacketCount) {
        const size_t rangeIdx = it->second;
        const int64_t packetDelta =
            static_cast<int64_t>(newPacketCount) - static_cast<int64_t>(oldPacketCount);

        ClPrint(
            amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
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
  return hipSuccess;  // Node not in any batch
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

  // Find the segment batch for this segment ID using O(1) map lookup
  auto segBatchIt = segmentBatches_.find(segmentId);
  if (segBatchIt == segmentBatches_.end()) {
    return hipSuccess; // Segment not found
  }

  auto& segBatch = segBatchIt->second;

  // Search only within this segment's packet batches
  for (auto& packetBatch : segBatch.packet_batches) {
    auto it = packetBatch.nodeToRangeIndex.find(node);
    if (it != packetBatch.nodeToRangeIndex.end()) {
      // Found the batch containing this node - update enabled state
      packetBatch.setEnabled(node, isEnabled);
      return hipSuccess;
    }
  }
  return hipSuccess;
}

// ================================================================================================

void GraphExec::DecrementRefCount(cl_event event, cl_int command_exec_status, void* user_data) {
  GraphExec* graphExec = reinterpret_cast<GraphExec*>(user_data);
  graphExec->release();
}

// ================================================================================================
void GraphExec::AssignStreamsToSegments(
    const std::vector<int>& segments_at_level,
    hip::Stream* launch_stream,
    const std::vector<hip::Stream*>& streams,
    std::unordered_map<int, hip::Stream*>& segment_to_stream) {

  // Assign streams to segments at this level using round-robin
  for (size_t idx = 0; idx < segments_at_level.size(); ++idx) {
    int segment_id = segments_at_level[idx];
    const auto& segment = segments_[segment_id];

    // Determine device ID for this segment from its first node
    int segment_device_id = launch_stream->DeviceId();
    if (!segment.nodes.empty() && segment.first_node != nullptr) {
      segment_device_id = segment.first_node->GetDeviceId();
    }

    hip::Stream* assigned_stream = nullptr;

    // Use collision-handled streams if provided (single-device case)
    if (!streams.empty()) {
      // Round-robin across the collision-handled streams
      size_t stream_idx = idx % streams.size();
      assigned_stream = streams[stream_idx];
    } else if (parallel_streams_.find(segment_device_id) != parallel_streams_.end() &&
               !parallel_streams_[segment_device_id].empty()) {
      // Multi-device case: Use device-aware stream selection from parallel_streams_
      const auto& device_streams = parallel_streams_[segment_device_id];
      size_t stream_idx = idx % (device_streams.size() + 1);
      assigned_stream = (stream_idx == 0) ? launch_stream : device_streams[stream_idx - 1];
    } else {
      // Fallback to launch stream if no parallel streams available
      assigned_stream = launch_stream;
    }

    segment_to_stream[segment_id] = assigned_stream;
  }
}

// ================================================================================================
amd::Command* GraphExec::EnqueueSegmentedGraph(hip::Stream* launch_stream,
                                               const std::vector<hip::Stream*>& streams,
                                               hipError_t* out_status) {
  hipError_t status = hipSuccess;
  if (out_status != nullptr) {
    *out_status = hipSuccess;
  }

  // Lambda to create and enqueue a marker with wait list
  auto enqueueMarker = [](hip::Stream* stream, const amd::Command::EventWaitList& wait_list) {
    auto marker = new amd::Marker(*stream, true, wait_list);
    // Marker is only for dependency, no need to flush caches.
    marker->setCommandEntryScope(amd::Device::kCacheStateIgnore);
    marker->enqueue();
    marker->release();
  };

  // Map to track which stream each segment uses - MUST persist across all levels
  // so we can look up streams for dependencies from previous levels
  std::unordered_map<int, hip::Stream*> segment_to_stream;
  // Map to track the last enqueued command for each segment for dependency tracking
  // This is critical for handling cross-level dependencies with stream reuse
  std::unordered_map<int, amd::Command*> segment_last_command;
  // Set of segment IDs that have already been explicitly synchronized to the
  // launch_stream via an earlier cross-stream wait marker. These segments can be
  // safely excluded from the final "sync all streams to launch_stream" step to
  // avoid inserting redundant markers.
  std::unordered_set<int> segments_synced_to_launch;

  // Process segments level by level using the pre-calculated max_dependency_level_
  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto level_it = segments_per_level_.find(level);
    if (level_it == segments_per_level_.end()) {
      continue;
    }

    const auto& segments_at_level = level_it->second;

    // Assign streams to segments at this level
    AssignStreamsToSegments(segments_at_level, launch_stream, streams, segment_to_stream);

    // Process each segment at this level
    for (int segment_id : segments_at_level) {
      const auto& segment = segments_[segment_id];
      hip::Stream* current_stream = segment_to_stream[segment_id];

      // Handle dependencies: add wait markers if dependent segments are on different streams
      // Look up the specific command for each dependency segment
      amd::Command::EventWaitList wait_list;
      for (int dep_segment_id : segment.segment_ids_dependencies) {
        // Dependencies are present in the segment_to_stream and segment_last_command map
        auto stream_it = segment_to_stream.find(dep_segment_id);
        if (stream_it == segment_to_stream.end()) {
          continue;
        }
        hip::Stream* dep_stream = stream_it->second;

        // Need to wait if dependency is on a different stream
        if (dep_stream != current_stream) {
          auto cmd_it = segment_last_command.find(dep_segment_id);
          if (cmd_it != segment_last_command.end() && cmd_it->second != nullptr) {
            // Retain command before adding to wait list for proper lifetime management
            cmd_it->second->retain();
            wait_list.push_back(cmd_it->second);
            if (current_stream == launch_stream) {
              segments_synced_to_launch.insert(dep_segment_id);
            }
          }
        }
      }

      // If there are cross-stream dependencies, insert a marker to wait
      if (!wait_list.empty()) {
        enqueueMarker(current_stream, wait_list);
        // Release our retains - marker has its own retain on wait list events
        for (auto* cmd : wait_list) {
          cmd->release();
        }
      }

      // Create accumulate command for this segment
      amd::AccumulateCommand* accumulate = new amd::AccumulateCommand(*current_stream, {}, nullptr);

      // Enqueue this segment using the helper function
      status = EnqueueSegment(segment, current_stream, accumulate);

      if (status != hipSuccess) {
        accumulate->release();
        // Clean up any previously enqueued commands
        for (auto& pair : segment_last_command) {
          if (pair.second != nullptr) {
            pair.second->release();
          }
        }
        if (out_status != nullptr) {
          *out_status = status;
        }
        return nullptr;
      }

      // Do not release as this is released at the end
      accumulate->enqueue();

      segment_last_command[segment_id] = accumulate;
    }
  }

  // Synchronize all streams with work back to launch_stream
  // Build a map of stream to last command by collecting from the highest-level segment on each
  // stream This is critical because unordered_map iteration order is undefined, so we must
  // explicitly track dependency levels to ensure we wait on the last command (highest level) on
  // each stream
  std::unordered_map<hip::Stream*, amd::Command*> stream_last_command_map;
  std::unordered_map<hip::Stream*, int> stream_max_level; // Track max dependency level per stream

  for (const auto& pair : segment_last_command) {
    int seg_id = pair.first;

    auto stream_it = segment_to_stream.find(seg_id);
    if (segments_synced_to_launch.find(seg_id) != segments_synced_to_launch.end() ||
        stream_it == segment_to_stream.end()) {
      continue;
    }

    amd::Command* cmd = pair.second;
    hip::Stream* stream = stream_it->second;
    int seg_dependency_level = segments_[seg_id].dependency_level;

    // Only update if this segment is at a strictly higher level
    // Using strict > ensures deterministic behavior when multiple segments
    // are at the same level on the same stream
    auto level_it = stream_max_level.find(stream);
    if (level_it == stream_max_level.end() || seg_dependency_level > level_it->second) {
      stream_max_level[stream] = seg_dependency_level;
      stream_last_command_map[stream] = cmd;
    }
  }

  amd::Command::EventWaitList final_wait_list;
  for (const auto& pair : stream_last_command_map) {
    hip::Stream* stream = pair.first;
    amd::Command* last_cmd = pair.second;

    // Sync all streams except the launch_stream itself
    if (stream != launch_stream && last_cmd != nullptr) {
      // Retain commands before adding to wait list since marker will retain them
      // and we'll release them later in cleanup
      last_cmd->retain();
      final_wait_list.push_back(last_cmd);
    }
  }

  // If there are other streams with work, sync them back to launch_stream
  if (!final_wait_list.empty()) {
    enqueueMarker(launch_stream, final_wait_list);
  }

  // Release the extra retains for commands in final_wait_list
  // (marker has its own retain, we release ours)
  for (auto* cmd : final_wait_list) {
    if (cmd != nullptr) {
      cmd->release();
    }
  }

  // Get the last command enqueued on the launch_stream for parent dependency tracking
  // This is to prevent release in cleanup loop, this determines graph execution completion
  amd::Command* last_command = nullptr;
  auto launch_stream_it = stream_last_command_map.find(launch_stream);
  if (launch_stream_it != stream_last_command_map.end()) {
    last_command = launch_stream_it->second;
    // Find the segment that produced this command and remove it from cleanup
    for (auto it = segment_last_command.begin(); it != segment_last_command.end(); ) {
      if (it->second == last_command) {
        it = segment_last_command.erase(it);
        break;
      } else {
        ++it;
      }
    }
  }

  // Release all other enqueued accumulate commands
  for (auto& pair : segment_last_command) {
    if (pair.second != nullptr) {
      pair.second->release();
    }
  }

  if (out_status != nullptr) {
    *out_status = status;
  }
  return last_command;
}

// ================================================================================================
// Graph segment to queue dispatch matching
hipError_t GraphExec::EnqueueSegment(const Segment& segment, hip::Stream* stream,
                                     amd::AccumulateCommand* accumulate) {
  hipError_t status = hipSuccess;

  // Find the SegmentBatch for this segment using O(1) map lookup
  SegmentBatch* segBatch = nullptr;
  auto segBatchIt = segmentBatches_.find(segment.id);
  if (segBatchIt != segmentBatches_.end()) {
    segBatch = &segBatchIt->second;
  }

  size_t batchIndex = 0;

  // Handle child graph segments - recursively enqueue the entire child graph
  if (segment.child_graph_ptr != nullptr) {
    auto childGraphExec = dynamic_cast<GraphExec*>(segment.child_graph_ptr);
    if (childGraphExec != nullptr) {
      // Child graphs share the same kernel arg manager as parent (for packet capture)
      if (childGraphExec->GetKernelArgManager() == nullptr) {
        auto kernArgMgr = GetKernelArgManager();
        if (kernArgMgr != nullptr) {
          kernArgMgr->retain();  // Increment ref count for child's reference
          childGraphExec->SetKernelArgManager(kernArgMgr);
        }
      }

      // Recursively enqueue the child graph with its own dependency tracking
      // Child graphs use their own parallel_streams_, so pass empty vector
      hipError_t child_status = hipSuccess;
      amd::Command* child_last_cmd = childGraphExec->EnqueueSegmentedGraph(
          stream, {}, &child_status);

      if (child_status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] EnqueueSegment: Failed to enqueue child graph, status=%d",
                child_status);
        return child_status;
      }

      // Child graph's work is already enqueued to the stream
      // The returned last command tracks completion - release our reference
      if (child_last_cmd != nullptr) {
        child_last_cmd->release();
      }
    }

    // Child graph segment has no regular nodes to process
    return hipSuccess;
  }

  // Process all nodes in this segment
  for (size_t i = 0; i < segment.nodes.size(); ++i) {
    auto& node = segment.nodes[i];
    if (DEBUG_HIP_GRAPH_DOT_PRINT) {
      node->stream_id_ = stream->GetStreamId();
      node->hw_queue_id_ = stream->getQueueID();
    }
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
        if (DEBUG_HIP_GRAPH_DOT_PRINT) {
          for(int j = i; j < i + packetBatch.nodeRanges.size(); j++) {
            segment.nodes[j]->stream_id_ = stream->GetStreamId();
            segment.nodes[j]->hw_queue_id_ = stream->getQueueID();
          }
        }
        // Skip all consecutive captured nodes that belong to this batch
        i += packetBatch.nodeRanges.size() - 1;  // -1 because loop will increment
        ++batchIndex;
      }
      if (DEBUG_HIP_GRAPH_DOT_PRINT) {
        node->hw_queue_id_ = node->GetQueue()->getQueueID();
      }
    }
  }

  return status;
}

// ================================================================================================
void GraphExec::UpdateStreams(hip::Stream* launch_stream) {
  int devId = launch_stream->vdev()->device().index();
  // Clear any previous stream assignments
  streams_.clear();
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
bool Graph::RunOneNode(Node node) {
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
      } else {
        // Release nodes that were enqueued on the same stream, since they are not included in the
        // wait list. Their references were retained for all outgoing edges.
        for (auto command : depNode->GetCommands()) {
          command->release();
        }
      }
    } else {
      node->SetWait(false);
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
    // If a wait was requested, then process the list
    if (node->GetWait() && !waitList.empty()) {
      node->UpdateEventWaitLists(waitList);
    }
    // Start the execution
    node->EnqueueCommands(node->GetQueue());
  }
  // Release commands of dependency nodes that were included in the wait list after enqueue
  for (auto dep : wait_order_) {
    if (dep != nullptr) {
      // Add all commands in the wait list
      if (dep->GetType() != hipGraphNodeTypeGraph) {
        for (auto command : dep->GetCommands()) {
          command->release();
        }
      }
    }
  }
  // Assign the launch ID of the submmitted node
  // This is also applied to childGraphs to prevent them from being reprocessed
  node->launch_id_ = current_id_++;
  uint32_t i = 0;
  // Execute the nodes in the edges list
  for (auto edge : node->GetEdges()) {
    // Don't wait in the nodes, executed on the same streams and if it has just one dependency
    bool wait =
        ((i < DEBUG_HIP_FORCE_GRAPH_QUEUES) || (edge->GetDependencies().size() > 1)) ? true : false;
    edge->SetWait(wait);
    i++;
    // Retain the current node for all its outgoing edges.
    // Each edge will include this node in its waitlist and release it after their commands are
    // enqueued.
    for (auto command : node->GetCommands()) {
      command->retain();
    }
  }
  if (node->GetEdges().size() == 0) {
    // Add a leaf node into the list for a wait.
    // Always use the last node, since it's the latest for the particular queue
    leafs_[node->stream_id_] = node;
    // An extra retain is needed for the leaves in order to be able to later enqueue a marker
    // on the app stream that has these commands in the waitlist.
    if (node->GetType() != hipGraphNodeTypeGraph) {
      for (auto command : node->GetCommands()) {
        command->retain();
      }
    }
  }

  node->SetWait(false);
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
    start_marker->enqueue();
    start_marker->release();
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
        start_marker->enqueue();
        start_marker->release();
      }
    }
    last_command->release();
  }

  // Run all commands in the graph
  for (auto node : GetTopoOrder()) {
    node->launch_id_ = -1;
    if (!RunOneNode(node)) {
      return false;
    }
  }
  wait_list.clear();
  // Check if the graph has multiple leaf nodes
  for (uint32_t i = 0; i < DEBUG_HIP_FORCE_GRAPH_QUEUES; ++i) {
    if ((leafs_[i] != nullptr) && (leafs_[i]->GetType() != hipGraphNodeTypeGraph)) {
      // Add all commands in the wait list
      for (auto command : leafs_[i]->GetCommands()) {
        if (base_stream != i) {
          wait_list.push_back(command);
        } else {
          command->release();
        }
      }
    }
  }
  // Wait for leafs in the graph's app stream
  if (wait_list.size() > 0) {
    auto end_marker = new amd::Marker(*streams_[base_stream], true, wait_list);
    end_marker->enqueue();
    end_marker->release();
    for (auto command : wait_list) {
      command->release();
    }
  }

  return true;
}

hipError_t ihipGraphDebugDotPrint(hip::Graph* graph, const char* path, unsigned int flags);

// ================================================================================================
hipError_t GraphExec::Run(hip::Stream* launch_stream) {
  hipError_t status = hipSuccess;

  // Get the first node based on scheduling mode
  Node firstNode = nullptr;
  if (use_segment_scheduling_ && !segments_.empty() && !segments_[0].nodes.empty()) {
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

  if (use_segment_scheduling_ && instantiateDeviceId_ == launch_stream->DeviceId()) {
    // If the graph has kernels that does device side allocation,  during packet capture, heap is
    // allocated because heap pointer has to be added to the AQL packet, and initialized during
    // graph launch.
    static bool initialized = false;
    // Todo: Hidden heap initialization is done only for single device graph
    if (!initialized && HasHiddenHeap()) {
      launch_stream->vdev()->HiddenHeapInit();
      initialized = true;
    }
    // Update streams for the graph execution only if launch stream changed
    if (lastLaunchStream_ != launch_stream) {
      UpdateStreams(launch_stream);
      lastLaunchStream_ = launch_stream;
    }
    amd::Command* last_cmd = nullptr;
    if (max_streams_dev_.size() == 1) {
      // Single-device: pass collision-handled streams_ to EnqueueSegmentedGraph
      last_cmd = EnqueueSegmentedGraph(launch_stream, streams_, &status);
    } else {
      // Multi-device: pass empty vector, will use parallel_streams_ internally
      last_cmd = EnqueueSegmentedGraph(launch_stream, {}, &status);
    }

    // Release the last command as we don't need to track it for top-level graph execution
    if (last_cmd != nullptr) {
      last_cmd->release();
    }
  } else if (max_streams_ == 1 && instantiateDeviceId_ != launch_stream->DeviceId()) {
    for (int i = 0; i < topoOrder_.size(); i++) {
      topoOrder_[i]->SetStream(launch_stream);
      status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
      topoOrder_[i]->EnqueueCommands(launch_stream);
    }
  } else {
    // Update streams for the graph execution only if launch stream changed
    if (lastLaunchStream_ != launch_stream) {
      UpdateStreams(launch_stream);
      lastLaunchStream_ = launch_stream;
    }
    // Execute all nodes in the graph
    if (!RunNodes()) {
      LogError("Failed to launch nodes!");
      return hipErrorOutOfMemory;
    }
  }
  if (DEBUG_HIP_GRAPH_DOT_PRINT == 2 && !graph_dumped_) {
    graph_dumped_ = true;
    std::string filename =
        "graph_" + std::to_string(amd::Os::getProcessId()) + "_dot_print_launch_1";
    hipError_t status = ihipGraphDebugDotPrint(this, filename.c_str(), 0);
    if (status == hipSuccess) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "[hipGraph] graph dump:%s", filename.c_str());
    }
  }
  this->retain();
  amd::Command* CallbackCommand = new amd::Marker(*launch_stream, kMarkerDisableFlush, {});
  // we may not need to flush any caches.
  CallbackCommand->setCommandEntryScope(amd::Device::kCacheStateIgnore);
  amd::Event& event = CallbackCommand->event();
  constexpr bool kBlocking = false;
  if (!event.setCallback(CL_COMPLETE, GraphExec::DecrementRefCount, this, kBlocking)) {
    this->release();
    CallbackCommand->release();
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
#if defined(ATI_ARCH_X86)
      _mm_sfence();
#else
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
      *sentinel_ptr = kSentinel;
#if defined(ATI_ARCH_X86)
      _mm_mfence();
#else
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
      kSentinel = *sentinel_ptr;
      (void)kSentinel; // Suppress unused variable warning
    }
  }
}
}  // namespace hip
