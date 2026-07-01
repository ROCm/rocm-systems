/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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

std::atomic<int> GraphNode::nextID{0};
std::atomic<int> Graph::nextID{0};
std::unordered_set<GraphNode*> GraphNode::nodeSet_;
// Guards global node set
amd::Monitor GraphNode::nodeSetLock_{};
std::unordered_set<Graph*> Graph::graphSet_;
// Guards global graph set
amd::Monitor Graph::graphSetLock_{};
std::unordered_set<GraphExec*> GraphExec::graphExecSet_;
// Guards global exec graph set
// we have graphExec object as part of child graph and we need recursive lock
std::recursive_mutex GraphExec::graphExecSetLock_;
// Serialize the creation of internal streams from multiple threads, ensuring that each stream is
// mapped to different HSA queues.
std::recursive_mutex GraphExec::graphExecStreamCreateLock_;
std::shared_mutex GraphExec::graphExecTrimLock_;
std::unordered_set<UserObject*> UserObject::ObjectSet_;
// Guards global user object
amd::Monitor UserObject::UserObjectLock_{};
// Guards mem map add/remove against work thread
amd::Monitor GraphNode::WorkerThreadLock_{};

hipError_t GraphMemcpyNode1D::ValidateParams(void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  if (dst == nullptr || src == nullptr) {
      return hipErrorInvalidValue;
  }
  if (static_cast<uint32_t>(kind) > hipMemcpyDefault && kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObjectForCurrentDevice(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObjectForCurrentDevice(dst, dOffset);

  if ((srcMemory == nullptr) && (dstMemory != nullptr)) {  // host to device
    if ((kind != hipMemcpyHostToDevice) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {  // device to host
    if ((kind != hipMemcpyDeviceToHost) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  }

  if (srcMemory != nullptr || dstMemory != nullptr) {
    hip::Device* dev = hip::getCurrentDevice();
    if (dev == nullptr) {
      return hipErrorInvalidDevice;
    }
    amd::Device& amdDev = *dev->devices()[0];
    if (srcMemory != nullptr) {
      hipError_t status =
          ihipMemcpy_validate_memory(amdDev, srcMemory, count, sOffset, /*read_write*/ false);
      if (status != hipSuccess) {
        return status;
      }
    }
    if (dstMemory != nullptr) {
      hipError_t status =
          ihipMemcpy_validate_memory(amdDev, dstMemory, count, dOffset, /*read_write*/ true);
      if (status != hipSuccess) {
        return status;
      }
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
void Graph::ScheduleOneNode(Node start, int stream_id) {
  if (!start) return;

  // stack of pending nodes for DFS
  std::vector<Node> pending;
  pending.push_back(start);

  int sid = stream_id;

  while (!pending.empty()) {
    Node cur = pending.back();
    pending.pop_back();

    // Skip if already scheduled
    if (cur->stream_id_ != -1) {
      continue;
    }

    // Schedule current node on this branch's stream
    cur->stream_id_ = sid;

    max_streams_ = std::max(max_streams_, sid + 1);
    streams_dev_ids_[sid].insert(cur->dev_id_);

    // Process child graph separately, since, there is no connection
    if (cur->GetType() == hipGraphNodeTypeGraph) {
      auto cgn   = reinterpret_cast<hip::ChildGraphNode*>(cur);
      auto child = cgn->GetChildGraph();
      // Use same scheduling logic(classic or segment) as parent graph for child graph
      child->SetSegmentScheduling(use_segment_scheduling_);
      hipError_t status = child->ScheduleNodes();
      (void)status;
      max_streams_ = std::max(max_streams_, child->max_streams_);
    }

    const auto& edges = cur->GetEdges();
    bool end_of_branch = true;

    // To preserve left-to-right behavior, push siblings in reverse so the earlier
    // edges get processed first.
    for (int i = static_cast<int>(edges.size()) - 1; i >= 0; --i) {
      Node e = edges[static_cast<size_t>(i)];
      if (e->stream_id_ != -1) continue;
      pending.push_back(e);
      end_of_branch = false;
    }

    if (end_of_branch) {
      // Finished one depth traversal (one branch). Rotate for the next sibling/branch.
      sid = (sid + 1) % DEBUG_HIP_FORCE_GRAPH_QUEUES;
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

      // Use a set for O(1) duplicate detection instead of linear search on the vector
      std::unordered_set<int> dep_set(segment.segment_ids_dependencies.begin(),
                                      segment.segment_ids_dependencies.end());

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

          // Add dependency if not already present (O(1) lookup)
          if (dep_set.insert(dep_segment_id).second) {
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
void GraphExec::BuildSyncPlan() {
  // Clean up any prior barrier packets
  for (auto* p : sync_plan_.barrier_packets) { delete[] p; }

  sync_plan_.num_segments = static_cast<int>(segments_.size());
  sync_plan_.patch_list.clear();
  sync_plan_.barrier_packets.clear();
  sync_plan_.leaf_segment_ids.clear();
  sync_plan_.seg_to_hw_event.assign(segments_.size(), -1);
  sync_plan_.num_hw_events = 0;

  auto* device = g_devices[instantiateDeviceId_]->devices()[0];

  // PASS 0: Barrier-ROI collapse. When multi-stream's cross-stream sync would not
  // pay for the overlap it unlocks (see ShouldCollapseToSingleStream), fold every
  // segment onto stream 0. EnqueueSegmentedGraph resolves stream 0 to the launch
  // stream (streams_[0]), so the whole graph runs on that one in-order queue: the
  // subsequent passes then emit zero cross-stream barriers and zero completion
  // signals, and graph completion is observed through the launch stream itself.
  // Init() reads collapsed_to_single_stream_ to create just one stream per device.
  collapsed_to_single_stream_ = false;
  if (ShouldCollapseToSingleStream()) {
    for (auto& seg : segments_) {
      seg.stream_id = 0;
      seg.needs_completion_signal = false;
    }
    collapsed_to_single_stream_ = true;
  }

  // PASS 1: Assign a compact HW-event slot only to segments whose completion
  // signal is consumed — cross-device/stream successor, or leaf when
  // leaf-sync is required. Same-stream successors are ordered by the
  // in-order queue and need no signal.
  // needs_completion_signal is pre-computed by PrecomputeStreamAssignment()
  // using the same criteria, so we reuse it directly.
  for (size_t i = 0; i < segments_.size(); ++i) {
    if (segments_[i].needs_completion_signal) {
      sync_plan_.seg_to_hw_event[i] = sync_plan_.num_hw_events++;
    }
  }

  // PASS 2: Eliminate redundant cross-stream dependency barriers.
  //
  // Segments dispatch level-by-level and, within a level, round-robin across the
  // stream pool (see PrecomputeStreamAssignment / EnqueueSegmentedGraph), so
  // multiple segments can land on the same (device, stream). Because each stream
  // is an in-order HW queue, once an earlier-dispatched segment on a stream has
  // waited (via a barrier) for a producer's completion signal, every later
  // segment on that same stream is already ordered after that producer finishes
  // and must NOT re-wait for it. e.g. two level-1 segments that share the same
  // level-0 dependencies and land on the same stream: only the first needs the
  // dep barrier; the second inherits the ordering for free.
  //
  // effective_barrier_deps[seg.id] holds the minimal set of cross-stream
  // producers each segment must still wait on. Default to the full cross-stream
  // set (safe for any segment not reached by the dispatch-order walk below),
  // then reduce. NOTE: a segment's id equals its position in segments_ (ids are
  // assigned sequentially at creation and pushed in order). The whole sync plan
  // relies on this (e.g. segments_[dep_id]), so assert it once and index
  // effective_barrier_deps by segment.id consistently below.
  std::vector<std::vector<int>> effective_barrier_deps(segments_.size());
  for (size_t i = 0; i < segments_.size(); ++i) {
    const auto& seg = segments_[i];
    assert(seg.id == static_cast<int>(i) &&
           "segment.id must equal its index in segments_");
    for (int dep_id : seg.segment_ids_dependencies) {
      if (dep_id < 0 || dep_id >= static_cast<int>(segments_.size())) continue;
      const auto& dep_seg = segments_[dep_id];
      if (dep_seg.dev_id != seg.dev_id || dep_seg.stream_id != seg.stream_id) {
        effective_barrier_deps[seg.id].push_back(dep_id);
      }
    }
  }
  {
    // Per-stream set of producer segments already waited on, keyed by
    // (dev_id, stream_id) packed into one 64-bit value. Walk segments in the
    // exact dispatch order used by EnqueueSegmentedGraph.
    std::unordered_map<uint64_t, std::unordered_set<int>> stream_waited_deps;
    auto stream_key = [](int dev_id, int stream_id) -> uint64_t {
      return (static_cast<uint64_t>(static_cast<uint32_t>(dev_id)) << 32) |
             static_cast<uint32_t>(stream_id);
    };

    for (int level = 0; level <= max_dependency_level_; ++level) {
      auto level_it = segments_per_level_.find(level);
      if (level_it == segments_per_level_.end()) continue;

      for (int seg_id : level_it->second) {
        if (seg_id < 0 || seg_id >= static_cast<int>(segments_.size())) continue;
        const auto& seg = segments_[seg_id];
        auto& waited = stream_waited_deps[stream_key(seg.dev_id, seg.stream_id)];

        std::vector<int>& reduced = effective_barrier_deps[seg_id];
        reduced.clear();
        for (int dep_id : seg.segment_ids_dependencies) {
          if (dep_id < 0 || dep_id >= static_cast<int>(segments_.size())) continue;
          const auto& dep_seg = segments_[dep_id];
          // Same-stream/device deps are ordered by the in-order queue already.
          if (dep_seg.dev_id == seg.dev_id && dep_seg.stream_id == seg.stream_id) {
            continue;
          }
          // Cross-stream dep: emit a wait only if no earlier same-stream segment
          // has waited for this producer yet. insert() returns true on first add.
          if (waited.insert(dep_id).second) {
            reduced.push_back(dep_id);
          }
        }
      }
    }
  }

  // Barrier packets are sentinel-marked with nullptr in dispatchKernelNames so that
  // activity.cpp can distinguish them from kernel/blit dispatch packets (which use "" or a
  // real name).  This avoids the empty-string ambiguity that caused the last kernel node to
  // be dropped when a copy/blit node also contributed an empty-string entry.
  static const std::string* const kBarrierKernelNamePtr = nullptr;

  // PASS 3: Materialize barrier packets and patch entries using the compact
  // hw_event slot indices computed in PASS 1.
  for (const auto& segment : segments_) {
    // Minimal cross-stream/device dependency set computed in PASS 2 (redundant
    // same-stream barriers already removed).
    const std::vector<int>& barrier_dep_indices = effective_barrier_deps[segment.id];

    auto segBatchIt = segmentBatches_.find(segment.id);
    if (segBatchIt == segmentBatches_.end()) {
      continue;
    }

    auto& segBatch = segBatchIt->second;

    // Ensure at least one PacketBatch exists for barrier placement
    if (segBatch.packet_batches.empty()) {
      segBatch.packet_batches.emplace_back();
    }

    auto& firstBatch = segBatch.packet_batches[0];

    // Prepend barrier packets for segments with dependencies.
    // Optimization: when there is exactly 1 dependency and the first captured
    // packet is an ext kernel dispatch, embed the dep_signal directly into
    // that packet instead of creating a separate barrier.
    if (!barrier_dep_indices.empty()) {
      int num_deps = static_cast<int>(barrier_dep_indices.size());
      bool use_ext_dep = false;
      if (num_deps == 1 && !firstBatch.dispatchPackets.empty()) {
        const uint8_t* pkt = firstBatch.dispatchPackets[0];
        uint16_t first_hdr;
        memcpy(&first_hdr, pkt, sizeof(first_hdr));
        constexpr uint16_t kPktTypeMask = 0xFF;
        constexpr uint16_t kVendorSpecificType = 0;
        constexpr uint8_t kExtKernelDispatchFormat = 3;
        uint8_t amd_format = pkt[2];
        use_ext_dep = ((first_hdr & kPktTypeMask) == kVendorSpecificType)
                      && (first_hdr != 0)
                      && (amd_format == kExtKernelDispatchFormat);
      }

      if (use_ext_dep) {
        uint8_t* first_dispatch = firstBatch.dispatchPackets[0];
        // hw_event_index uses the compact slot; dep producer always has one (PASS 1).
        sync_plan_.patch_list.push_back(
            {first_dispatch, nullptr,
             sync_plan_.seg_to_hw_event[barrier_dep_indices[0]],
             amd::Device::HwEventPatch::kExtDispatchDepSignal});
      } else {
        int barrier_count = (num_deps + 4) / 5;

        for (int b = 0; b < barrier_count; ++b) {
          uint8_t* barrier_pkt = device->CreateBarrierPacket();
          sync_plan_.barrier_packets.push_back(barrier_pkt);

          int start_dep = b * 5;
          int end_dep = std::min(start_dep + 5, num_deps);
          for (int d = start_dep; d < end_dep; ++d) {
            sync_plan_.patch_list.push_back(
                {barrier_pkt, nullptr,
                 sync_plan_.seg_to_hw_event[barrier_dep_indices[d]],
                 d - start_dep});
          }

          firstBatch.dispatchPackets.insert(firstBatch.dispatchPackets.begin(), barrier_pkt);
          firstBatch.dispatchKernelNames.insert(firstBatch.dispatchKernelNames.begin(),
                                                kBarrierKernelNamePtr);
          firstBatch.dispatchMetadataPackets.insert(
              firstBatch.dispatchMetadataPackets.begin(), nullptr);
        }

        // nodeRanges[i].startIndex was recorded before barrier packets were prepended.
        // Update all node range indices in firstBatch to account for the inserted barriers.
        for (auto& nodeRange : firstBatch.nodeRanges) {
          nodeRange.startIndex += static_cast<size_t>(barrier_count);
        }
      }
    }

    bool last_node_uncaptured = segBatch.has_uncaptured_nodes &&
        !segment.nodes.empty() && !segBatch.node_capture_status.back();

    // hw_slot >= 0 => some consumer observes this signal (set by PASS 1).
    // Otherwise skip both the completion barrier packet and its patch entry.
    const int hw_slot = sync_plan_.seg_to_hw_event[segment.id];
    const bool completion_signal_needed = (hw_slot >= 0);

    auto& lastBatch = segBatch.packet_batches.back();
    if (last_node_uncaptured && completion_signal_needed) {
      uint8_t* completion_barrier = device->CreateBarrierPacket();
      sync_plan_.barrier_packets.push_back(completion_barrier);

      lastBatch.dispatchPackets.push_back(completion_barrier);
      lastBatch.dispatchKernelNames.push_back(kBarrierKernelNamePtr);
      lastBatch.dispatchMetadataPackets.push_back(nullptr);

      sync_plan_.patch_list.push_back(
          {completion_barrier, nullptr, hw_slot,
           amd::Device::HwEventPatch::kCompletionSignal});
    } else if (!lastBatch.dispatchPackets.empty() && completion_signal_needed) {
      // Safe to patch the last kernel dispatch directly
      uint8_t* last_pkt = lastBatch.dispatchPackets.back();
      sync_plan_.patch_list.push_back(
          {last_pkt, nullptr, hw_slot,
           amd::Device::HwEventPatch::kCompletionSignal});
    }

    if (segment.segment_ids_edges.empty()) {
      sync_plan_.leaf_segment_ids.push_back(segment.id);
    }
  }

  // Create the per-graph HW event signal pool once at instantiate time
  // (single-threaded here) and pre-create the signals, so the launch hot path
  // only pops a ready set and patches it — never creating signals.
  if (signalManager_ == nullptr) {
    signalManager_ = new GraphSignalManager();
  }
  if (sync_plan_.num_hw_events > 0) {
    // Pre-create a few sets to cover a small amount of launch overlap; the pool
    // grows on demand if more launches are concurrently in flight.
    constexpr int kPrecreatedSets = 16;
    signalManager_->Prepopulate(device, sync_plan_.num_hw_events, kPrecreatedSets);
  }

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
          "[hipGraph] BuildSyncPlan: %d segments, %zu barrier packets, %d completion signals",
          sync_plan_.num_segments, sync_plan_.barrier_packets.size(), sync_plan_.num_hw_events);
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
    FindPathsDFS(root, current_path, visited, graph_paths);
  }
  return graph_paths;
}

// ================================================================================================
void Graph::FindPathsDFS(Node start, std::vector<Node>& current_path,
                         std::unordered_set<unsigned int>& visited,
                         hip::Graph::GraphExecutionPaths& graph_paths) {
  // Lambda to save current path as a HierarchicalPath
  auto savePath = [&graph_paths](std::vector<Node> path, int device_id,
                                  Node child_node = nullptr, int child_index = -1) {
    hip::Graph::HierarchicalPath h_path;
    h_path.nodes = std::move(path);
    h_path.device_id = device_id;
    h_path.child_graph_node = child_node;
    h_path.child_graph_paths_index = child_index;
    graph_paths.paths.push_back(std::move(h_path));
  };

  if (!start) return;

  // Stack of nodes to process.
  std::vector<Node> st;
  st.push_back(start);

  while (!st.empty()) {
    Node node = st.back();
    st.pop_back();
    if (!node) continue;

    // Check if already visited
    if (visited.find(node->GetID()) != visited.end()) {
      // Save any remaining path (treat visited as a branch end)
      if (!current_path.empty()) {
        int dev = current_path.back()->GetDeviceId();
        savePath(std::move(current_path), dev);
        current_path.clear();
      }
      continue;
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
        savePath(std::move(current_path), prev_device_id);
        current_path.clear();
      }
    }

    // Handle child graph nodes specially
    if (node->GetType() == hipGraphNodeTypeGraph) {
      // Save path before child graph node (if any)
      if (!current_path.empty()) {
        int dev = current_path.back()->GetDeviceId();
        savePath(std::move(current_path), dev);
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
          childGraph->FindPathsDFS(child_root, child_current_path, child_visited,
                                   child_graph_exec_paths);
        }

        // Store the child graph paths
        int child_graph_index = static_cast<int>(graph_paths.child_graph_paths.size());
        graph_paths.child_graph_paths.push_back(std::move(child_graph_exec_paths));

        // Create a path containing just the child graph node
        std::vector<Node> child_node_path = {childGraphNode};
        savePath(child_node_path, current_device_id, childGraphNode, child_graph_index);
      }

      // Clear current path and continue with edges from the child graph node
      current_path.clear();
      const auto& edges = node->GetEdges();
      for (int i = static_cast<int>(edges.size()) - 1; i >= 0; --i) {
        st.push_back(edges[static_cast<size_t>(i)]);
      }

      continue;
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
        Node saved_join_node = nullptr;

        // For join nodes, save path without the join node itself
        // For fork nodes, save the complete path
        if (is_join) {
          saved_join_node = current_path.back();
          current_path.pop_back();
        }

        if (!current_path.empty()) {
          int dev = current_path.back()->GetDeviceId();
          savePath(current_path, dev);
        }
        current_path.clear();

        // For nodes that are both fork and join, save them as their own segment
        if (saved_join_node != nullptr && is_fork) {
          std::vector<Node> fork_join_segment = {saved_join_node};
          savePath(std::move(fork_join_segment), saved_join_node->GetDeviceId());
        }

        // Put the join node back in current_path for further traversal
        // But not if it's also a fork node, because we'll traverse branches separately
        if (saved_join_node != nullptr && !is_fork) {
          current_path.push_back(saved_join_node);
        }
      }

      // Traverse each branch until it hits a join
      for (int i = static_cast<int>(edges.size()) - 1; i >= 0; --i) {
        st.push_back(edges[static_cast<size_t>(i)]);
      }
    } else if (edges.size() == 1) {
      // Single edge - continue on same path
      st.push_back(edges[0]);
    }

    // Save any remaining path (handles leaf nodes and leaf join nodes)
    if (!current_path.empty() && edges.size() == 0) {
      int dev = current_path.back()->GetDeviceId();
      savePath(std::move(current_path), dev);
      current_path.clear();
    }
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
    segment.dev_id = h_path.device_id;
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
  std::scoped_lock lock(graphExecSetLock_);
  if (graphExecSet_.find(pGraphExec) == graphExecSet_.end()) {
    return false;
  }
  return true;
}

// ================================================================================================
hipError_t GraphExec::CreateStreams(uint32_t num_streams, int devId) {
  std::scoped_lock lock(graphExecStreamCreateLock_);

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

  // num_streams is already capped by Init() but guard here defensively.
  // For the instantiation device one slot is occupied by the launch stream,
  // so create one fewer extra stream. Other devices use all slots as parallel streams.
  uint32_t capped = std::min(num_streams, DEBUG_HIP_FORCE_GRAPH_QUEUES);
  uint32_t max_streams = (devId == instantiateDeviceId_ && capped > 0) ? capped - 1 : capped;
  if (max_streams == 0) {
    return hipSuccess;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Creating %u parallel streams for device %d",
          max_streams, devId);
  parallel_streams_[devId].reserve(max_streams);
  // Track queue IDs already assigned to earlier internal streams so each new
  // stream avoids colliding with them at creation time.
  std::unordered_set<uint64_t> used_qids;
  for (uint32_t i = 0; i < max_streams; ++i) {
    auto stream = new hip::Stream(g_devices[devId], hip::Stream::Priority::Normal,
                                  hipStreamNonBlocking);

    if (!stream->Create()) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to create stream %u for device %d",
              i, devId);
      hip::Stream::Destroy(stream);
      for (auto& created_stream : parallel_streams_[devId]) {
        created_stream->vdev()->UnpinQueue();
        hip::Stream::Destroy(created_stream);
      }
      parallel_streams_[devId].clear();
      return hipErrorOutOfMemory;
    }

    // Pin the queue so dynamic queue management won't release it between launches
    stream->vdev()->PinQueue();
    // Acquire a queue that doesn't collide with previously created internal streams.
    // On the first stream (used_qids empty) this is a normal acquisition.
    if (!used_qids.empty()) {
      stream->vdev()->ReacquireQueueExcluding(used_qids);
    }
    used_qids.insert(stream->getQueueID());

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
void GraphExec::FindStreamsReqPerDevForSegments() {
  // For packet engine mode: analyze segments to determine stream requirements per device
  // We need to track the maximum number of concurrent segments per device at any level

  max_streams_dev_.clear();
  std::unordered_map<int, int> streams_per_dev_at_level;
  std::vector<GraphExec*> graphs_to_process{this};

  while (!graphs_to_process.empty()) {
    GraphExec* graphExec = graphs_to_process.back();
    graphs_to_process.pop_back();
    if (graphExec == nullptr) {
      continue;
    }

    if (graphExec != this && graphExec->instantiateDeviceId_ == -1) {
      graphExec->instantiateDeviceId_ = instantiateDeviceId_;
      static_cast<amd::ReferenceCountedObject*>(g_devices[instantiateDeviceId_])->retain();
    }

    for (const auto& [level, segment_ids] : graphExec->segments_per_level_) {
      streams_per_dev_at_level.clear();

      // Count segments per device at this level
      for (int segment_id : segment_ids) {
        if (segment_id >= 0 && segment_id < static_cast<int>(graphExec->segments_.size())) {
          const auto& segment = graphExec->segments_[segment_id];

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

    for (const auto& segment : graphExec->segments_) {
      if (segment.child_graph_ptr != nullptr) {
        auto childGraphExec = dynamic_cast<GraphExec*>(segment.child_graph_ptr);
        if (childGraphExec != nullptr) {
          graphs_to_process.push_back(childGraphExec);
        }
      }
    }
  }
}

// ================================================================================================
void GraphExec::PrecomputeStreamAssignment() {
  // max_streams_dev_ holds the raw parallelism count per device as computed by
  // FindStreamsReqPerDev[ForSegments]() and capped in Init(). CreateStreams() handles
  // the -1 adjustment for the instantiation device internally, so the value here
  // represents the total stream pool size for every device uniformly.
  auto getPoolSize = [&](int dev_id) -> size_t {
    auto it = max_streams_dev_.find(dev_id);
    return (it != max_streams_dev_.end() && it->second > 0)
               ? static_cast<size_t>(it->second) : 1;
  };

  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto it = segments_per_level_.find(level);
    if (it == segments_per_level_.end()) continue;

    // Per-device round-robin counters, reset per level so parallel segments on
    // the same device spread evenly across that device's stream pool.
    std::unordered_map<int, size_t> dev_idx;

    for (int seg_id : it->second) {
      if (seg_id >= 0 && seg_id < static_cast<int>(segments_.size())) {
        auto& seg = segments_[seg_id];
        seg.stream_id = static_cast<int>(dev_idx[seg.dev_id]++ % getPoolSize(seg.dev_id));
      }
    }
  }

  ComputeCompletionSignalFlags();
}

// ================================================================================================
void GraphExec::ComputeCompletionSignalFlags() {
  const bool leaf_sync_required = IsLeafNodeSyncRequired();
  for (auto& seg : segments_) {
    seg.needs_completion_signal = false;
    if (seg.segment_ids_edges.empty()) {
      // Leaf segments need a completion signal so EnqueueSegmentedGraph can
      // sync them back to the launch stream via graph_accumulate dep_signals.
      if (leaf_sync_required) {
        seg.needs_completion_signal = true;
      }
      continue;
    }
    for (int edge_id : seg.segment_ids_edges) {
      if (edge_id >= 0 && edge_id < static_cast<int>(segments_.size())) {
        const auto& edge_seg = segments_[edge_id];
        // Signal needed if downstream segment is on a different stream OR a
        // different device — both cases require explicit HW synchronization.
        if (edge_seg.dev_id != seg.dev_id || edge_seg.stream_id != seg.stream_id) {
          seg.needs_completion_signal = true;
          break;
        }
      }
    }
  }
}

// ================================================================================================
// Barrier-ROI heuristic.
//
// Multi-stream segment scheduling only pays off when the device-side overlap it
// unlocks exceeds the cost of the cross-stream barriers/signals it requires. For
// launch-overhead-bound graphs (many tiny kernels, near-serial dependencies) the
// barriers have a high overhead: collapsing every segment onto one in-order stream
// removes them entirely and is measurably faster on both the launch and the
// instantiate path.
//
// Both sides are estimated in cheap structural units, with no device timing:
//   * barrier_est   = number of barrier *packets* multi-stream would emit. This
//                     mirrors BuildSyncPlan's materialization rather than the raw
//                     dependency count: a single barrier packet resolves up to 5
//                     dependencies ((deps + 4) / 5 packets), and a lone
//                     dependency is folded into the segment's ext kernel-dispatch
//                     packet with no separate barrier at all. PrecomputeStream-
//                     Assignment() has already run by the time this is called, so
//                     only *cross-stream/device* deps are counted (same-stream
//                     deps are ordered by the in-order queue and emit no barrier),
//                     matching what PASS 2/PASS 3 of BuildSyncPlan actually do.
//   * signal_est    = number of completion *signals* multi-stream would emit.
//                     Reuses each segment's needs_completion_signal flag (set by
//                     PrecomputeStreamAssignment with the same cross-stream/leaf
//                     criteria BuildSyncPlan uses), so producers whose consumers
//                     share their stream are not over-counted. Signals are real
//                     per-launch host cost (signal-pool acquire + packet patch)
//                     that the barrier count alone misses, so they belong in the
//                     ROI denominator alongside barriers.
//   * parallel_slack = total work - critical-path work. The work that is
//                      genuinely off the longest dependency chain and could
//                      therefore overlap on another stream. Work is measured in
//                      machine-occupancy passes, not node count: each kernel
//                      weighs ceil(launch_threads / machine_threads) (>=1), so a
//                      kernel that fills the GPU once is 1 unit and one needing N
//                      passes is N. Non-kernel nodes and sub-machine launches are
//                      1 unit, preserving the original node-count behaviour for
//                      the small launch-bound kernels the gate targets. This is
//                      what keeps two *long-running* independent kernels multi-
//                      stream (their slack outgrows the threshold) while tiny
//                      independent kernels (e.g. PyFR's stubs) still collapse.
//
// Collapse when parallel_slack < min_overlap * (barrier_est + signal_est): i.e.
// keep multi-stream only when each unit of cross-stream sync overhead buys at
// least min_overlap nodes of overlappable work. Folding collapse onto the launch
// stream removes both barriers and signals (it runs inline, 0/0), so the full
// sync cost is what multi-stream genuinely pays over collapse. Tunable via
// DEBUG_HIP_GRAPH_MIN_OVERLAP; 0 disables the gate.
bool GraphExec::ShouldCollapseToSingleStream() const {
  const uint32_t min_overlap = DEBUG_HIP_GRAPH_MIN_OVERLAP;
  if (min_overlap == 0) return false;            // gate disabled
  if (segments_.size() < 2) return false;        // nothing to parallelize

  // Collapse folds every segment onto stream 0, which EnqueueSegmentedGraph
  // resolves to the launch stream (streams_[0]). That only works when all
  // segments live on the same device; a multi-device graph would mis-route its
  // off-device segments onto the launch stream, so never collapse one.
  const int dev0 = segments_.front().dev_id;
  for (const auto& seg : segments_) {
    if (seg.dev_id != dev0) return false;
  }

  // Machine concurrent-thread capacity, used to convert a kernel's launch size
  // into whole occupancy passes (the per-node work weight below). Falls back to
  // node-count weighting (every node == 1) if the device info is unavailable.
  size_t machine_threads = 0;
  if (dev0 >= 0) {
    const auto& dinfo = g_devices[dev0]->devices()[0]->info();
    machine_threads = static_cast<size_t>(dinfo.maxComputeUnits_) * dinfo.maxThreadsPerCU_;
  }
  // Per-node work proxy: kernels are weighted by ceil(threads / machine_threads)
  // — i.e. how many full-machine occupancy passes the launch needs — so a launch
  // that fills the GPU once counts as 1, a launch needing N passes counts as N.
  // Sub-machine launches (and all non-kernel nodes) stay at 1, which preserves
  // the original node-count behaviour (and the min_overlap scale) for the small,
  // launch-bound kernels the gate originally targeted. NOTE: this is occupancy-
  // bound only; a small-grid kernel with a long internal loop is underweighted.
  auto node_work = [machine_threads](Node n) -> size_t {
    if (n == nullptr || n->GetType() != hipGraphNodeTypeKernel) return 1;
    const size_t threads = static_cast<GraphKernelNode*>(n)->GetLaunchThreadCount();
    if (threads == 0 || machine_threads == 0) return 1;
    return std::max<size_t>(1, (threads + machine_threads - 1) / machine_threads);
  };

  size_t barrier_est = 0;  // barrier packets multi-stream would emit (see header)
  size_t signal_est = 0;   // completion signals multi-stream would emit (see header)
  size_t total_work = 0;   // sum of per-node work over all segments
  std::vector<size_t> seg_work(segments_.size(), 0);
  for (size_t i = 0; i < segments_.size(); ++i) {
    const auto& seg = segments_[i];
    // Count only cross-stream/device dependencies. Same-stream deps are ordered
    // by the in-order queue and never materialize a barrier, so including them
    // would overstate multi-stream's cost and bias the gate toward collapse.
    // PrecomputeStreamAssignment() has already assigned stream_id by the time
    // this runs, so mirror the exact cross-stream filter PASS 2/PASS 3 use.
    size_t cross_deps = 0;
    for (int dep_id : seg.segment_ids_dependencies) {
      if (dep_id < 0 || dep_id >= static_cast<int>(segments_.size())) continue;
      const auto& dep_seg = segments_[dep_id];
      if (dep_seg.dev_id != seg.dev_id || dep_seg.stream_id != seg.stream_id) {
        ++cross_deps;
      }
    }
    // cross_deps == 1 is embedded into the ext kernel-dispatch packet (no
    // separate barrier); cross_deps >= 2 needs ceil(cross_deps / 5) barriers.
    if (cross_deps >= 2) {
      barrier_est += (cross_deps + 4) / 5;
    }
    // A segment emits a completion signal exactly when PrecomputeStreamAssignment
    // flagged it (cross-stream/device consumer, or leaf when leaf-join back to
    // the launch stream is required). Reuse that flag instead of counting every
    // segment with downstream edges, which over-counts same-stream producers.
    if (seg.needs_completion_signal) {
      ++signal_est;
    }
    for (Node n : seg.nodes) seg_work[i] += node_work(n);
    total_work += seg_work[i];
  }
  // Total per-launch cross-stream overhead multi-stream pays on the host path.
  const size_t sync_cost = barrier_est + signal_est;
  if (sync_cost == 0) return false;  // no barriers/signals => multi-stream is free

  // Critical path (in work units) via DP over segments in dependency-level order:
  // cp[seg] = work(seg) + max(cp[dep]); a dep always sits at a strictly lower level.
  std::vector<size_t> cp(segments_.size(), 0);
  size_t critical_path_work = 0;
  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto it = segments_per_level_.find(level);
    if (it == segments_per_level_.end()) continue;
    for (int seg_id : it->second) {
      if (seg_id < 0 || seg_id >= static_cast<int>(segments_.size())) continue;
      const auto& seg = segments_[seg_id];
      size_t best_dep = 0;
      for (int dep_id : seg.segment_ids_dependencies) {
        if (dep_id < 0 || dep_id >= static_cast<int>(segments_.size())) continue;
        best_dep = std::max(best_dep, cp[dep_id]);
      }
      cp[seg_id] = best_dep + seg_work[seg_id];
      critical_path_work = std::max(critical_path_work, cp[seg_id]);
    }
  }

  // parallel_slack = overlappable work off the critical path. Weighting by work
  // (not node count) keeps genuinely parallel long-running kernels multi-stream:
  // their slack scales with launch size and outgrows min_overlap * sync_cost,
  // while tiny launch-bound kernels still collapse.
  const size_t parallel_slack =
      (total_work > critical_path_work) ? (total_work - critical_path_work) : 0;
  const bool collapse = parallel_slack < static_cast<size_t>(min_overlap) * sync_cost;

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
          "[hipGraph] Single-stream gate: work=%zu critical_path=%zu slack=%zu "
          "barrier_packets~=%zu signals~=%zu sync_cost=%zu min_overlap=%u -> %s",
          total_work, critical_path_work, parallel_slack, barrier_est, signal_est,
          sync_cost, min_overlap,
          collapse ? "collapse to single stream" : "keep multi-stream");
  return collapse;
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

    // Cap per-device stream counts to the hardware queue limit. PrecomputeStreamAssignment()
    // reads max_streams_dev_ to assign segment stream ids, so both must see the capped values.
    for (auto& [dev_id, count] : max_streams_dev_) {
      count = std::min(count, static_cast<int>(DEBUG_HIP_FORCE_GRAPH_QUEUES));
    }

    if (!use_segment_scheduling_) {
      // Classic scheduling has no BuildSyncPlan collapse pass, so create streams now.
      for (auto const& [dev_id, num_streams] : max_streams_dev_) {
        if (num_streams > 0) {
          status = CreateStreams(num_streams, dev_id);
          if (status != hipSuccess) {
            return status;
          }
        }
      }
    }
  }

  if (use_segment_scheduling_) {
    // Pre-compute stream assignment before packet capture so that BuildSyncPlan
    // (called inside CaptureAQLPackets) can see each segment's stream_id and
    // skip same-stream dependency barriers.
    PrecomputeStreamAssignment();

    // For graph nodes capture AQL packets to dispatch them directly during graph launch.
    // BuildSyncPlan (inside) runs the barrier-ROI collapse pass, which may fold the
    // graph onto a single stream per device.
    status = CaptureAQLPackets();
    if (status != hipSuccess) {
      return status;
    }

    // Create parallel streams now (still at instantiate time, never lazily at launch),
    // sized to the final post-collapse assignment: one stream per device when the
    // collapse pass fired, otherwise the capped multi-stream counts.
    if (collapsed_to_single_stream_) {
      for (auto& [dev_id, count] : max_streams_dev_) {
        count = 1;
      }
    }
    uint32_t total_streams = 0;
    for (auto const& [dev_id, count] : max_streams_dev_) {
      total_streams += static_cast<uint32_t>(std::max(count, 0));
    }
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
            "[hipGraph] Init: %zu device(s), %u total stream(s) (per-device cap: %u)",
            max_streams_dev_.size(), total_streams, DEBUG_HIP_FORCE_GRAPH_QUEUES);
    for (auto const& [dev_id, num_streams] : max_streams_dev_) {
      if (num_streams > 0) {
        status = CreateStreams(num_streams, dev_id);
        if (status != hipSuccess) {
          return status;
        }
      }
    }
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
  filteredCacheValid = false;
}

// ================================================================================================
// Rebuild cached filtered lists of enabled packets.
// Barrier packets (prepended/appended by BuildSyncPlan) live in dispatchPackets
// but are not tracked in nodeRanges.  A linear scan with a per-index enabled
// bitmap preserves them while filtering out disabled node packets.
// ================================================================================================
void GraphExec::PacketBatch::rebuildFilteredLists(
    std::vector<amd::Device::HwEventPatch>& patch_list) {
  if (filteredCacheValid) {
    return;
  }

  // Default every packet to enabled; mark disabled-node packets false.
  std::vector<bool> packetEnabled(dispatchPackets.size(), true);
  for (const auto& range : nodeRanges) {
    if (!range.enabled) {
      for (size_t j = 0; j < range.packetCount; ++j) {
        packetEnabled[range.startIndex + j] = false;
      }
    }
  }

  enabledPackets.clear();
  enabledKernelNames.clear();
  filteredFlatPacketData.clear();
  filteredValidPacketFullHeaders.clear();
  filteredFlatMetadataData.clear();

  enabledPackets.reserve(dispatchPackets.size());
  enabledKernelNames.reserve(dispatchPackets.size());
  filteredFlatPacketData.reserve(dispatchPackets.size() * kAqlPktSize);
  filteredValidPacketFullHeaders.reserve(dispatchPackets.size());

  const bool hasMetadata = !dispatchMetadataPackets.empty();

  // packet pointer -> index in the filtered flat buffer, built during the
  // single pass below so patch_list resolution is O(patches) not O(p*n).
  std::unordered_map<const void*, size_t> packetToFilteredIndex;

  for (size_t i = 0; i < dispatchPackets.size(); ++i) {
    if (packetEnabled[i]) {
      size_t filteredIdx = enabledPackets.size();
      enabledPackets.push_back(dispatchPackets[i]);
      enabledKernelNames.push_back(dispatchKernelNames[i]);
      appendPacketToFlatBuffer(dispatchPackets[i], filteredFlatPacketData,
                               filteredValidPacketFullHeaders);
      // Append corresponding metadata slot (zero-filled for barriers)
      if (hasMetadata) {
        size_t metaOff = filteredFlatMetadataData.size();
        filteredFlatMetadataData.resize(metaOff + kMetadataPktSize, 0);
        if (i < dispatchMetadataPackets.size() && dispatchMetadataPackets[i] != nullptr) {
          std::memcpy(filteredFlatMetadataData.data() + metaOff,
                      dispatchMetadataPackets[i], kMetadataPktSize);
        }
      }
      packetToFilteredIndex[dispatchPackets[i]] = filteredIdx;
    }
  }

  // Re-point flat_packet pointers in patch_list into filteredFlatPacketData.
  for (auto& patch : patch_list) {
    auto it = packetToFilteredIndex.find(patch.packet);
    if (it != packetToFilteredIndex.end()) {
      patch.flat_packet =
          filteredFlatPacketData.data() + it->second * kAqlPktSize;
    }
  }

  filteredCacheValid = true;
}

// ================================================================================================
// Restore flat_packet pointers in patch_list back to flatPacketData.
// Called when all nodes are re-enabled (disabledNodeCount == 0) so that
// ApplyHwEventPatches writes into the buffer the dispatch path will use.
// ================================================================================================
void GraphExec::PacketBatch::restorePatchListPointers(
    std::vector<amd::Device::HwEventPatch>& patch_list) {
  for (auto& patch : patch_list) {
    for (size_t i = 0; i < dispatchPackets.size(); ++i) {
      if (patch.packet == dispatchPackets[i]) {
        patch.flat_packet = flatPacketData.data() + i * kAqlPktSize;
        break;
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
    // Child-graph segments: create a SegmentBatch with leading + trailing empty batches
    // so BuildSyncPlan can prepend dep barriers and append a completion barrier
    if (segment.child_graph_ptr != nullptr) {
      auto [it, inserted] = segmentBatches_.emplace(segment.id, segment.id);
      auto& childSegBatch = it->second;
      childSegBatch.node_capture_status.resize(segment.nodes.size(), false);
      childSegBatch.has_uncaptured_nodes = true;
      childSegBatch.packet_batches.emplace_back();  // leading: dep barriers
      childSegBatch.packet_batches.emplace_back();  // trailing: completion barrier
      continue;
    }

    // Always create a SegmentBatch for every non-child-graph segment
    auto [it, inserted] = segmentBatches_.emplace(segment.id, segment.id);
    // Initialize node_capture_status for this segment
    auto& currentSegBatch = it->second;
    currentSegBatch.node_capture_status.resize(segment.nodes.size(), false);

    bool first_node_is_uncaptured = !segment.nodes.empty() &&
                                    !segment.nodes[0]->GraphCaptureEnabled();

    // Leading empty batch: gives BuildSyncPlan a slot to prepend the cross-dep
    // BARRIER_AND so it physically precedes the uncaptured node's GPU commands.
    if (first_node_is_uncaptured) {
      currentSegBatch.packet_batches.emplace_back();
    }

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

      if (node->GraphCaptureEnabled()) {
        // Start of a new batch
        PacketBatch newBatch;

        // Collect packets from consecutive captured nodes
        size_t j = i;
        while (j < segment.nodes.size() && segment.nodes[j]->GraphCaptureEnabled()) {
          auto& currentNode = segment.nodes[j];
          // Empty nodes are pure dependency points — no GPU commands or markers.
          // Cross-stream ordering is handled by BuildSyncPlan barrier packets.
          if (currentNode->GetType() == hipGraphNodeTypeEmpty) {
            const size_t rangeIndex = newBatch.nodeRanges.size();
            newBatch.nodeRanges.push_back({newBatch.dispatchPackets.size(), 0, true});
            newBatch.nodeToRangeIndex[currentNode] = rangeIndex;
            currentSegBatch.node_capture_status[j] = true;
            ++j;
            continue;
          }
          // Capture packets for this node
          std::vector<uint8_t*> nodePackets;
          std::vector<const std::string*> nodeKernelNames;
          std::vector<uint8_t*> nodeMetadataPackets;
          status = currentNode->CaptureAndFormPacket(GetKernelArgManager(), &nodePackets,
                                                     &nodeKernelNames, &nodeMetadataPackets);

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
          newBatch.dispatchMetadataPackets.reserve(startIndex + packetCount);

          // Add to dispatch lists (initially all enabled)
          newBatch.dispatchPackets.insert(newBatch.dispatchPackets.end(), nodePackets.begin(),
                                          nodePackets.end());
          newBatch.dispatchKernelNames.insert(newBatch.dispatchKernelNames.end(),
                                              nodeKernelNames.begin(), nodeKernelNames.end());
          newBatch.dispatchMetadataPackets.insert(newBatch.dispatchMetadataPackets.end(),
                                                  nodeMetadataPackets.begin(),
                                                  nodeMetadataPackets.end());

          // Store node mapping with range info
          newBatch.nodeRanges.push_back({startIndex, packetCount, true});
          newBatch.nodeToRangeIndex[currentNode] = rangeIndex;

          // Mark this node as successfully captured
          currentSegBatch.node_capture_status[j] = true;
          ++j;
        }

        // Add the batch if it contains packets or captured zero-packet nodes (e.g. EMPTY).
        if (!newBatch.dispatchPackets.empty() || !newBatch.nodeRanges.empty()) {
          currentSegBatch.packet_batches.push_back(std::move(newBatch));
        }
        i = j - 1;  // for-loop will ++i to j
      } else {
        // Non-capturable node
        currentSegBatch.has_uncaptured_nodes = true;
        currentSegBatch.node_capture_status[i] = false;
      }
    }

    // Trailing empty batch: separate slot for the completion barrier so it
    // cannot fire before the uncaptured last node's commands finish.
    bool last_node_uncaptured = currentSegBatch.has_uncaptured_nodes &&
        !segment.nodes.empty() && !currentSegBatch.node_capture_status.back();
    if (last_node_uncaptured) {
      currentSegBatch.packet_batches.emplace_back();
    }
  }

  // Recursively process child graphs to capture their packets
  for (const auto& segment : segments_) {
    if (segment.child_graph_ptr != nullptr) {
      auto childGraphExec = dynamic_cast<GraphExec*>(segment.child_graph_ptr);
      if (childGraphExec != nullptr) {
        // Propagate instantiation device ID so BuildSyncPlan can
        // access the device for barrier packet creation.
        // Retain balances the release in ~Graph destructor.
        if (childGraphExec->instantiateDeviceId_ == -1) {
          childGraphExec->instantiateDeviceId_ = instantiateDeviceId_;
          static_cast<amd::ReferenceCountedObject*>(
              g_devices[instantiateDeviceId_])->retain();
        }

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
          ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                  "[hipGraph] Child graph packet capture failed for child graph in segment, "
                  "status=%d",
                  status);
          return status;
        }
      }
    }
  }

  // Build sync plan now that all segment batches are populated --
  // prepends barrier packets and generates the patch list
  BuildSyncPlan();

  // Build flat buffers once now that all dispatchPackets are finalized
  // (capture populated them, BuildSyncPlan may have prepended/appended barriers).
  // Also build a map from dispatchPacket pointer -> flat buffer pointer so
  // ApplyHwEventPatches can patch flatPacketData directly at launch time.
  std::unordered_map<const uint8_t*, uint8_t*> pktToFlat;
  for (auto& [seg_id, segBatch] : segmentBatches_) {
    for (auto& batch : segBatch.packet_batches) {
      if (!batch.dispatchPackets.empty()) {
        batch.rebuildFlatBuffer();
        for (size_t i = 0; i < batch.dispatchPackets.size(); ++i) {
          pktToFlat[batch.dispatchPackets[i]] =
              batch.flatPacketData.data() + i * PacketBatch::kAqlPktSize;
        }
      }
    }
  }

  // Resolve flat_packet pointers for all HwEventPatches
  for (auto& patch : sync_plan_.patch_list) {
    auto it = pktToFlat.find(patch.packet);
    if (it != pktToFlat.end()) {
      patch.flat_packet = it->second;
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
      std::vector<const std::string*> newKernelNames;
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
              static_cast<size_t>(packetDelta), nullptr);
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
        uint8_t* oldPkt = packetBatch.dispatchPackets[packetIndex];
        uint8_t* newPkt = newPackets[i];
        packetBatch.dispatchPackets[packetIndex] = newPkt;
        packetBatch.dispatchKernelNames[packetIndex] = newKernelNames[i];

        // Update SyncPlan patch list to point to the new packet
        // ApplyHwEventPatches patches the correct packet at launch time.
        if (oldPkt != newPkt) {
          for (auto& patch : sync_plan_.patch_list) {
            if (patch.packet == oldPkt) {
              patch.packet = newPkt;
            }
          }
        }
      }
      // Rebuild the flat buffer immediately so the next dispatch uses updated packets.
      // The flat buffer always represents the full packet sequence; the dispatch path
      // independently skips it when any nodes are disabled (disabledNodeCount != 0).
      packetBatch.rebuildFlatBuffer();

      // Refresh flat_packet pointers in the patch list since rebuildFlatBuffer
      // reallocated flatPacketData, invalidating previous flat_packet pointers.
      for (auto& patch : sync_plan_.patch_list) {
        for (size_t pi = 0; pi < packetBatch.dispatchPackets.size(); ++pi) {
          if (patch.packet == packetBatch.dispatchPackets[pi]) {
            patch.flat_packet = packetBatch.flatPacketData.data() + pi * PacketBatch::kAqlPktSize;
            break;
          }
        }
      }
      return hipSuccess;
    }
  }
  return hipSuccess;  // Node not in any batch
}

// ================================================================================================
// Append one 64-byte AQL packet to a flat buffer: copies the body, saves the original full_header
// and invalidates the header.
void GraphExec::PacketBatch::appendPacketToFlatBuffer(const uint8_t* pkt_raw,
                                                      std::vector<uint8_t>& flatData,
                                                      std::vector<uint32_t>& fullHeaders) {
  static constexpr size_t kSigOff = 56;
  const size_t baseOff = flatData.size();
  flatData.insert(flatData.end(), pkt_raw, pkt_raw + kAqlPktSize);
  uint8_t* dst = flatData.data() + baseOff;
  uint32_t fullHeader = 0;
  memcpy(&fullHeader, pkt_raw, sizeof(fullHeader));
  fullHeaders.push_back(fullHeader);
  // Set header to HSA_PACKET_TYPE_INVALID (type=1) so the GPU CP skips this
  // packet until the valid header is committed with release semantics during
  // dispatch. Using type=0 (VENDOR_SPECIFIC) would be a processable packet type
  // that the CP could attempt to execute with incomplete body data.
  static constexpr uint16_t kInvalidAqlHeader = 1;
       //HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  memcpy(dst, &kInvalidAqlHeader, sizeof(kInvalidAqlHeader));
  // Zero completion signal; ApplyHwEventPatches re-patches it directly via flat_packet pointers.
  memset(dst + kSigOff, 0, sizeof(uint64_t));
}

// ================================================================================================
// Rebuild the flat packet buffer from the current dispatchPackets contents.
void GraphExec::PacketBatch::rebuildFlatBuffer() {
  const size_t n = dispatchPackets.size();
  flatPacketData.clear();
  validPacketFullHeaders.clear();
  flatMetadataData.clear();
  filteredCacheValid = false;
  flatPacketData.reserve(n * kAqlPktSize);
  validPacketFullHeaders.reserve(n);
  for (const uint8_t* pkt_raw : dispatchPackets) {
    appendPacketToFlatBuffer(pkt_raw, flatPacketData, validPacketFullHeaders);
  }
  // Build flat metadata buffer (kMetadataPktSize per slot).
  // Entries without metadata (barriers) are zero-filled.
  if (!dispatchMetadataPackets.empty()) {
    flatMetadataData.resize(n * kMetadataPktSize, 0);
    for (size_t i = 0; i < n && i < dispatchMetadataPackets.size(); ++i) {
      if (dispatchMetadataPackets[i] != nullptr) {
        std::memcpy(flatMetadataData.data() + i * kMetadataPktSize,
                    dispatchMetadataPackets[i], kMetadataPktSize);
      }
    }
  }
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
      if (packetBatch.disabledNodeCount > 0) {
        // Eagerly rebuild filtered lists and re-resolve patch_list flat_packet
        // pointers so the launch path doesn't need to scan all segment batches.
        packetBatch.rebuildFilteredLists(sync_plan_.patch_list);
      } else {
        // All nodes re-enabled: restore flat_packet pointers back to
        // flatPacketData so ApplyHwEventPatches patches the correct buffer.
        packetBatch.restorePatchListPointers(sync_plan_.patch_list);
      }
      return hipSuccess;
    }
  }
  return hipSuccess;
}

// Carries the per-launch state needed by the completion callback: the graph
// whose refcount to drop, plus the signal set (and its device) to re-arm and
// return to the pool now that the launch's GPU work is done.
struct GraphLaunchCleanup {
  GraphExec* exec;
  amd::Device* device;
  std::vector<void*> signal_set;
};

void GraphExec::OnLaunchComplete(cl_event event, cl_int command_exec_status, void* user_data) {
  auto* cleanup = reinterpret_cast<GraphLaunchCleanup*>(user_data);
  GraphExec* graphExec = cleanup->exec;
  // Re-arm and recycle the launch's signals while the GraphExec (and thus its
  // signal pool) is still alive, then drop the launch's reference.
  if (graphExec->signalManager_ != nullptr && !cleanup->signal_set.empty()) {
    graphExec->signalManager_->ReleaseSet(cleanup->device, cleanup->signal_set);
  }
  delete cleanup;
  graphExec->release();
}

// ================================================================================================
amd::Command* GraphExec::EnqueueSegmentedGraph(hip::Stream* launch_stream,
                                               const std::vector<hip::Stream*>& streams,
                                               hipError_t* out_status,
                                               std::vector<void*>* out_signal_set) {
  hipError_t status = hipSuccess;
  if (out_status != nullptr) {
    *out_status = hipSuccess;
  }

  auto* device = g_devices[launch_stream->DeviceId()]->devices()[0];

  // Top-level launches recycle signals through the per-graph pool; the legacy
  // recursive child-graph path (out_signal_set == nullptr) creates them locally
  // and lets the AccumulateCommand destructor destroy them.
  const bool recycle = (out_signal_set != nullptr);

  std::vector<void*> segment_hw_events;
  if (sync_plan_.num_hw_events > 0) {
    const bool ok = recycle
        ? signalManager_->AcquireSet(device, sync_plan_.num_hw_events, segment_hw_events)
        : device->CreateHwEvents(sync_plan_.num_hw_events, segment_hw_events);
    if (!ok) {
      if (out_status != nullptr) {
        *out_status = hipErrorOutOfMemory;
      }
      return nullptr;
    }
  }

  // Apply pre-computed patches -- writes HW events directly into flatPacketData
  // via the flat_packet pointers resolved at instantiate time, so no rebuild needed.
  if (!sync_plan_.patch_list.empty()) {
    device->ApplyHwEventPatches(sync_plan_.patch_list, segment_hw_events);
  }

  // Resolve a segment's assigned hip::Stream* from its pre-computed stream_id.
  // streams is the collision-handled streams_ vector built by UpdateStreams.
  auto resolveSegmentStream = [&](const Segment& seg) -> hip::Stream* {
    if (!streams.empty()) {
      return streams[static_cast<size_t>(seg.stream_id) % streams.size()];
    }
    return launch_stream;
  };

  // Single AccumulateCommand on launch_stream manages all HW event lifetimes
  // and serves as the dispatch anchor for all segments across all streams.
  // Pass `this` as the kernel-names owner: the command borrows kernel-name
  // strings owned by this graph's nodes (via setKernelNamesRef during dispatch)
  // and reads them in ReportActivity() at completion, after OnLaunchComplete()
  // drops the launch's reference. Tying the GraphExec's lifetime to the command
  // keeps those strings valid through the report (no copies). We already hold a
  // launch reference here, so the retain in the constructor needs no trim lock.
  auto* graph_accumulate = new amd::AccumulateCommand(*launch_stream, {}, nullptr, this);

  // Register HW events with graph_accumulate so profiling can read them.
  for (auto& hw_event : segment_hw_events) {
    if (hw_event != nullptr) {
      graph_accumulate->addHwEvent(hw_event, device);
    }
  }

  // For the recycling (top-level) path, the pool owns the signals: hand the set
  // back to the caller, which forwards it to the completion callback
  // (OnLaunchComplete) that re-arms and returns it to the pool. Tell the
  // AccumulateCommand destructor not to destroy them. The legacy path keeps the
  // default (destructor destroys the locally created signals).
  if (recycle && !segment_hw_events.empty()) {
    graph_accumulate->setOwnsHwEvents(false);
    *out_signal_set = segment_hw_events;
  }

  // Process segments level by level
  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto level_it = segments_per_level_.find(level);
    if (level_it == segments_per_level_.end()) {
      continue;
    }

    const auto& segments_at_level = level_it->second;

    if (level == 0) {
      // Synchronize internal streams with launch stream's last command if available
      amd::Command* launch_last_cmd = launch_stream->getLastQueuedCommand(true);
      if (launch_last_cmd != nullptr) {
        amd::Command::EventWaitList launch_wait_list;
        launch_wait_list.push_back(launch_last_cmd);

        // For each segment at level 0, if it's on a different stream, add a wait marker
        for (int segment_id : segments_at_level) {
          hip::Stream* seg_stream = resolveSegmentStream(segments_[segment_id]);
          if (seg_stream != launch_stream) {
            auto marker = new amd::Marker(*seg_stream, true, launch_wait_list);
            if (marker != nullptr) {
              marker->enqueue();
              marker->release();
            }
          }
        }
        launch_last_cmd->release();
      }
    }

    // Dispatch each segment -- barriers are in the batch, signals are patched
    for (int segment_id : segments_at_level) {
      const auto& segment = segments_[segment_id];
      hip::Stream* current_stream = resolveSegmentStream(segment);

      status = EnqueueSegment(segment, current_stream, graph_accumulate);

      if (status != hipSuccess) {
        graph_accumulate->release();
        if (out_status != nullptr) {
          *out_status = status;
        }
        return nullptr;
      }
    }
  }

  // Sync parallel-stream leaves back to launch_stream via graph_accumulate's
  // dep_signal[]. Same-stream leaves rely on in-order queue semantics instead.
  if (IsLeafNodeSyncRequired()) {
    for (int seg_id : sync_plan_.leaf_segment_ids) {
      if (seg_id < 0 || seg_id >= static_cast<int>(segments_.size())) continue;
      hip::Stream* seg_stream = resolveSegmentStream(segments_[seg_id]);
      if (seg_stream == launch_stream) continue;
      int hw_slot = sync_plan_.seg_to_hw_event[seg_id];  // PASS 1 guarantees >= 0; guard defensively.
      if (hw_slot < 0 || hw_slot >= static_cast<int>(segment_hw_events.size())) continue;
      graph_accumulate->addDepHwEvent(segment_hw_events[hw_slot]);
    }
  }

  graph_accumulate->enqueue();

  if (out_status != nullptr) {
    *out_status = status;
  }
  return graph_accumulate;
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

  // Lambda to dispatch the current batch at batchIndex.
  // attach_signal=true asks the dispatcher to give the last packet a real
  // completion signal (via Barriers().ActiveSignal) so a downstream
  // uncaptured node — typically an SDMA memcpy on a different engine — can
  // wait on it directly through HwQueueTracker::WaitingSignal.
  auto dispatchCurrentBatch = [&](bool attach_signal = false) -> hipError_t {
    if (!segBatch || batchIndex >= segBatch->packet_batches.size()) {
      return hipSuccess;
    }
    auto& packetBatch = segBatch->packet_batches[batchIndex];
    if (packetBatch.dispatchPackets.empty()) {
      ++batchIndex;
      return hipSuccess;
    }

    const std::vector<const std::string*>* kernelNamesToDispatch;
    const std::vector<uint8_t>* flatData;
    const std::vector<uint32_t>* flatHdrs;
    const std::vector<uint8_t>* metaData = nullptr;

    if (packetBatch.disabledNodeCount == 0) {
      kernelNamesToDispatch = &packetBatch.dispatchKernelNames;
      flatData = &packetBatch.flatPacketData;
      flatHdrs = &packetBatch.validPacketFullHeaders;
      if (!packetBatch.flatMetadataData.empty()) {
        metaData = &packetBatch.flatMetadataData;
      }
    } else {
      // Guard against stale filtered buffers: rebuildFlatBuffer (called from
      // UpdateAQLPacket) invalidates the cache. This is a no-op when valid.
      packetBatch.rebuildFilteredLists(sync_plan_.patch_list);
      kernelNamesToDispatch = &packetBatch.enabledKernelNames;
      flatData = &packetBatch.filteredFlatPacketData;
      flatHdrs = &packetBatch.filteredValidPacketFullHeaders;
      if (!packetBatch.filteredFlatMetadataData.empty()) {
        metaData = &packetBatch.filteredFlatMetadataData;
      }
    }

    if (!flatData->empty()) {
      bool batchStatus = stream->vdev()->dispatchAqlPacketBatchFlat(
          *flatData, *flatHdrs, accumulate, attach_signal, kernelNamesToDispatch, true,
          false, metaData);
      if (!batchStatus) {
        return hipErrorUnknown;
      }
    }

    ++batchIndex;
    return hipSuccess;
  };

  // Handle child graph segments - recursively enqueue the entire child graph
  if (segment.child_graph_ptr != nullptr) {
    // Dispatch dependency barriers before child graph execution
    status = dispatchCurrentBatch();
    if (status != hipSuccess) return status;

    auto childGraphExec = dynamic_cast<GraphExec*>(segment.child_graph_ptr);
    if (childGraphExec != nullptr) {
      // Child graphs share the same kernel arg manager as parent (for packet capture)
      if (childGraphExec->GetKernelArgManager() == nullptr) {
        auto kernArgMgr = GetKernelArgManager();
        if (kernArgMgr != nullptr) {
          kernArgMgr->retain();
          childGraphExec->SetKernelArgManager(kernArgMgr);
        }
      }

      // Recursively enqueue the child graph with its own dependency tracking.
      // TODO: child graphs currently take the legacy create/destroy signal path
      // (out_signal_set == nullptr -> recycle == false), so their pre-created
      // signal pool (from the child's BuildSyncPlan/Prepopulate) sits unused and
      // they pay signal_create/destroy every launch. To pool child signals too,
      // pass an out_signal_set here and recycle it from the parent's
      // OnLaunchComplete (the parent's accumulate completion encloses the
      // child's work); the cleanup would carry per-pool (manager, set) pairs.
      hipError_t child_status = hipSuccess;
      amd::Command* child_last_cmd =
          childGraphExec->EnqueueSegmentedGraph(stream, {}, &child_status);

      if (child_status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] EnqueueSegment: Failed to enqueue child graph, status=%d",
                child_status);
        return child_status;
      }

      if (child_last_cmd != nullptr) {
        child_last_cmd->release();
      }
    }

    // Dispatch completion barrier after child graph — signals parent's HW event
    while (segBatch && batchIndex < segBatch->packet_batches.size()) {
      status = dispatchCurrentBatch();
      if (status != hipSuccess) return status;
    }

    return hipSuccess;
  }

  // Dispatch the leading batch (cross-dep barrier from BuildSyncPlan) before
  // the uncaptured first node so the AQL barrier precedes its GPU commands.
  if (segBatch && !segBatch->node_capture_status.empty() &&
      !segBatch->node_capture_status[0] &&
      batchIndex < segBatch->packet_batches.size()) {
    status = dispatchCurrentBatch();
    if (status != hipSuccess) return status;
  }

  // Process all nodes in this segment
  for (size_t i = 0; i < segment.nodes.size(); ++i) {
    auto& node = segment.nodes[i];

    if (segBatch && i < segBatch->node_capture_status.size() &&
        segBatch->node_capture_status[i]) {
      // Node was successfully captured - dispatch its batch
      if (segBatch && batchIndex < segBatch->packet_batches.size()) {
        auto& packetBatch = segBatch->packet_batches[batchIndex];
        if (DEBUG_HIP_GRAPH_DOT_PRINT) {
          for (size_t j = i; j < i + packetBatch.nodeRanges.size(); j++) {
            segment.nodes[j]->stream_id_ = stream->GetStreamId();
            segment.nodes[j]->hw_queue_id_ = stream->getQueueID();
          }
        }
        // Skip all consecutive captured nodes that belong to this batch
        i += packetBatch.nodeRanges.size() - 1;
        // Check if the next uncaptured node is an SDMA memcpy that needs handoff.
        // Only set sdma_follows if that's the case and the node will use SDMA.
        // Otherwise, staging-blit or other paths do not need the signal.
        bool sdma_follows = false;
        size_t next = i + 1;
        if (next < segment.nodes.size() &&
            next < segBatch->node_capture_status.size() &&
            !segBatch->node_capture_status[next] &&
            segment.nodes[next]->GetType() == hipGraphNodeTypeMemcpy) {
          auto* memcpyNode = dynamic_cast<GraphMemcpyNode*>(segment.nodes[next]);
          // Memcpy node types that don't derive from GraphMemcpyNode
          // (e.g. GraphDrvMemcpyNode) fall back to the conservative
          // "assume SDMA" behavior.
          sdma_follows = (memcpyNode == nullptr) || !memcpyNode->WillBypassSdmaEngine();
        }
        if (sdma_follows && !packetBatch.dispatchPackets.empty()) {
          stream->vdev()->addSystemScope();
        }
        status = dispatchCurrentBatch(sdma_follows);
        if (status != hipSuccess) return status;
      }
    } else {
      // Node doesn't support capture - execute individually. Upstream flat batches
      // hand off via attach_signal (SDMA memcpy) or BuildSyncPlan dep barriers;
      // no pre/post Markers needed around the uncaptured node.
      if (DEBUG_HIP_GRAPH_DOT_PRINT) {
        node->stream_id_ = stream->GetStreamId();
        node->hw_queue_id_ = stream->getQueueID();
      }
      node->SetStream(stream);
      status = node->CreateCommand(node->GetQueue());
      if (status != hipSuccess) return status;
      node->EnqueueCommands(stream);
    }
  }

  // Dispatch any remaining batches (e.g. trailing completion barrier for segments
  // with uncaptured nodes where the last node was non-captured)
  if (segBatch) {
    while (batchIndex < segBatch->packet_batches.size()) {
      status = dispatchCurrentBatch();
      if (status != hipSuccess) return status;
    }
  }

  return status;
}

// ================================================================================================
void GraphExec::UpdateStreams(hip::Stream* launch_stream) {
  int devId = launch_stream->vdev()->device().index();
  streams_.clear();
  streams_.push_back(launch_stream);
  if (parallel_streams_.find(devId) == parallel_streams_.end()) {
    // No parallel streams were created for this device
    return;
  }
  auto& parallel_streams = parallel_streams_[devId];

  // Collect queue IDs already in use, starting with the launch stream
  std::unordered_set<uint64_t> used_qids;
  used_qids.insert(launch_stream->getQueueID());

  for (auto stream : parallel_streams) {
    uint64_t qid = stream->getQueueID();
    if (used_qids.count(qid) > 0) {
      // Collision: this stream shares a HW queue with the launch stream or another
      // internal stream. Re-acquire a different queue, avoiding all used ones.
      if (stream->vdev()->ReacquireQueueExcluding(used_qids)) {
        qid = stream->getQueueID();
        ClPrint(amd::LOG_INFO, amd::LOG_CODE,
                "[hipGraph] Resolved queue collision: stream reassigned to queueID %lu", qid);
      } else {
        ClPrint(amd::LOG_WARNING, amd::LOG_CODE,
                "[hipGraph] Could not resolve queue collision for stream (best-effort)");
      }
    }
    used_qids.insert(qid);
    streams_.push_back(stream);
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
      // Child graph nodes may internally dispatch work on streams other
      // than their assigned stream_id_, so the same-stream in-order
      // assumption does not hold.  Always treat them as cross-stream deps.
      if (depNode->stream_id_ != node->stream_id_ ||
          depNode->GetType() == hipGraphNodeTypeGraph) {
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
      for (auto command : dep->GetCommands()) {
        waitList.push_back(command);
      }
    }
  }
  if (node->GetType() == hipGraphNodeTypeGraph) {
    // Process child graph separately, since, there is no connection
    auto child = reinterpret_cast<hip::ChildGraphNode*>(node)->GetChildGraph();
    if (!reinterpret_cast<hip::ChildGraphNode*>(node)->GetGraphCaptureStatus()) {
      child->RunNodes(node->stream_id_, &streams_, &waitList);
      // Store the child graph's completion command so that downstream
      // dependency handling can use node->GetCommands() directly,
      // instead of querying getLastQueuedCommand at dependency time
      // (which could return unrelated later work on the same stream).
      auto completion = streams_[node->stream_id_]->getLastQueuedCommand(true);
      if (completion != nullptr) {
        // Release any previously stored completion command (from prior launches)
        for (auto cmd : node->GetCommands()) {
          cmd->release();
        }
        node->GetCommands().clear();
        node->GetCommands().push_back(completion);
      }
    }
  } else {
    // Assing a stream to the current node
    node->SetStream(streams_);
    if (DEBUG_HIP_GRAPH_DOT_PRINT) {
      node->hw_queue_id_ = node->GetQueue()->getQueueID();
    }
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
      for (auto command : dep->GetCommands()) {
        command->release();
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
    // Child graph nodes now have completion commands stored via GetCommands(),
    // so they participate in the leaf retain/release cycle like regular nodes.
    for (auto command : node->GetCommands()) {
      command->retain();
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
    // For child graphs launched on a non-zero base_stream, the root nodes
    // are on stream 0 (roots_[0] is never set because scheduling always
    // assigns the first root to stream 0 and skips it in root recording).
    // Sync stream 0 with base_stream so the child's work waits for the
    // parent's dependencies.
    if (base_stream != 0) {
      auto start_marker = new amd::Marker(*streams_[0], true, wait_list);
      start_marker->enqueue();
      start_marker->release();
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
    if (leafs_[i] != nullptr) {
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

  // Retain under shared lock so hipDeviceGraphMemTrim's refcount check is accurate.
  // The lock blocks only while trim holds the exclusive (write) lock.
  {
    std::shared_lock<std::shared_mutex> trim_guard(graphExecTrimLock_);
    this->retain();
  }

  // Get the first node based on scheduling mode
  Node firstNode = nullptr;
  if (use_segment_scheduling_ && !segments_.empty() && !segments_[0].nodes.empty()) {
    firstNode = segments_[0].nodes[0];
  } else if (!topoOrder_.empty()) {
    firstNode = topoOrder_[0];
  }

  if (flags_ & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    if (firstNode != nullptr) {
      auto* parentGraph = firstNode->GetParentGraph();
      auto* pool = parentGraph->Device()->GetGraphMemoryPool();
      for (auto* node : topoOrder_) {
        if (node->GetType() == hipGraphNodeTypeMemAlloc) {
          static_cast<GraphMemAllocNode*>(node)->ReleaseCachedMapping(pool, launch_stream);
        }
      }
      parentGraph->FreeAllMemory(launch_stream);
      parentGraph->memalloc_nodes_ = 0;
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
      this->release();
      return hipErrorInvalidValue;
    }
  } else {
    repeatLaunch_ = true;
  }

  ClPrint(amd::LOG_DEBUG, amd::LOG_CODE, "GraphExec::Run max_streams: %d, on device: %d",
          max_streams_, launch_stream->DeviceId());

  // If the launch stream lost its HW queue due to dynamic queue management,
  // try to re-acquire the same one it used last time.
  // Then run collision detection to ensure graph-internal streams don't share
  // a HW queue with the launch stream.
  launch_stream->vdev()->SetPreferredQueue();
  launch_stream->vdev()->AcquireQueueWithPreference();
  UpdateStreams(launch_stream);

  // Signals borrowed from the per-graph pool for this launch (segmented path
  // only); handed to the completion callback to re-arm and return to the pool.
  std::vector<void*> launch_signal_set;

  // Command whose completion drives OnLaunchComplete. On the segmented path we
  // reuse the graph's own accumulate command instead of enqueuing a dedicated marker
  amd::Command* completion_cmd = nullptr;

  if (use_segment_scheduling_ && instantiateDeviceId_ == launch_stream->DeviceId()) {
    // If the graph has kernels that does device side allocation,  during packet capture, heap is
    // allocated because heap pointer has to be added to the AQL packet, and initialized during
    // graph launch.
    // Todo: Hidden heap initialization is done only for single device graph
    if (HasHiddenHeap() &&
        hiddenHeapInitializedDevices_.insert(launch_stream->DeviceId()).second) {
      launch_stream->vdev()->HiddenHeapInit();
    }
    amd::Command* last_cmd = nullptr;
    if (max_streams_dev_.size() == 1) {
      // Single-device: pass collision-handled streams_ to EnqueueSegmentedGraph
      last_cmd = EnqueueSegmentedGraph(launch_stream, streams_, &status, &launch_signal_set);
    } else {
      // Multi-device: pass empty vector, will use parallel_streams_ internally
      last_cmd = EnqueueSegmentedGraph(launch_stream, {}, &status, &launch_signal_set);
    }

    // Drive OnLaunchComplete off this command's completion (its leaf-sync deps
    // already imply all parallel work is done). Our reference is released after
    // the callback is registered below; the queue keeps it alive until done.
    completion_cmd = last_cmd;
  } else if (max_streams_ == 1 && instantiateDeviceId_ != launch_stream->DeviceId()) {
    for (int i = 0; i < topoOrder_.size(); i++) {
      topoOrder_[i]->SetStream(launch_stream);
      status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
      topoOrder_[i]->EnqueueCommands(launch_stream);
    }
  } else {
    // Execute all nodes in the graph
    if (!RunNodes()) {
      LogError("Failed to launch nodes!");
      this->release();
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
  // Register the refcount/recycle callback. Prefer the graph's own accumulate
  // command (segmented path); otherwise enqueue a lightweight marker just to
  // carry the callback.
  amd::Command* CallbackCommand = completion_cmd;
  const bool own_callback_cmd = (CallbackCommand == nullptr);
  if (own_callback_cmd) {
    CallbackCommand = new amd::Marker(*launch_stream, kMarkerDisableFlush, {});
    // we may not need to flush any caches.
    CallbackCommand->setCommandEntryScope(amd::Device::kCacheStateIgnore);
  }
  amd::Event& event = CallbackCommand->event();
  constexpr bool kBlocking = false;
  auto* cleanup = new GraphLaunchCleanup();
  cleanup->exec = this;
  cleanup->device = g_devices[launch_stream->DeviceId()]->devices()[0];
  cleanup->signal_set = std::move(launch_signal_set);
  if (!event.setCallback(CL_COMPLETE, GraphExec::OnLaunchComplete, cleanup, kBlocking)) {
    // setCallback essentially never fails, but if it does the launch's GPU work
    // is already queued (the accumulate is enqueued and was told not to destroy
    // its signals). Drain that work, then run the same completion handling the
    // callback would have (recycle the borrowed pooled signals + drop our
    // reference) so they are not leaked and the pool does not shrink.
    launch_stream->finish();
    OnLaunchComplete(nullptr, CL_COMPLETE, cleanup);
    CallbackCommand->release();
    return hipErrorInvalidHandle;
  }
  // The marker must be enqueued to run; the accumulate command is already
  // enqueued by EnqueueSegmentedGraph. Either way, release our reference: the
  // queue keeps the command alive until completion, when the callback fires.
  if (own_callback_cmd) {
    CallbackCommand->enqueue();
  }
  CallbackCommand->release();
  return status;
}

// ================================================================================================
GraphSignalManager::~GraphSignalManager() {
  // No launches can be in flight at this point (GraphExec refcount guarantees
  // it outlives all launches), so every set is back in the free pool.
  for (auto& dev_pool : free_sets_) {
    amd::Device* device = dev_pool.first;
    for (auto& set : dev_pool.second) {
      // Pooled signals rest armed (value 1); mark them idle before destroy so
      // ~ProfilingSignal does not block waiting on an armed-but-idle signal.
      device->QuiesceHwEvents(set);
      for (void* sig : set) {
        if (sig != nullptr) {
          // Pair with CreateHwEvents() so non-ROCm devices can hook teardown.
          device->DestroyHwEvent(sig);
        }
      }
    }
  }
  free_sets_.clear();
}

bool GraphSignalManager::Prepopulate(amd::Device* device, int count, int num_sets) {
  if (count <= 0 || num_sets <= 0) {
    return true;
  }
  std::lock_guard<std::mutex> lock(lock_);
  auto& pool = free_sets_[device];

  // BuildSyncPlan is re-runnable, so Prepopulate may be called more than once.
  // If a prior run sized the sets for a different segment count, those sets are
  // unusable -- destroy and rebuild. No launches are in flight at (re)instantiate
  // time, so every set for this device is present in the free pool here.
  if (!pool.empty() && static_cast<int>(pool.back().size()) != count) {
    for (auto& set : pool) {
      device->QuiesceHwEvents(set);
      for (void* sig : set) {
        if (sig != nullptr) {
          device->DestroyHwEvent(sig);
        }
      }
    }
    pool.clear();
  }

  // Top up to num_sets only; do not unconditionally append on every call, which
  // would grow the pool without bound across re-instantiations.
  for (int i = static_cast<int>(pool.size()); i < num_sets; ++i) {
    std::vector<void*> set;
    if (!device->CreateHwEvents(count, set)) {
      return false;
    }
    pool.push_back(std::move(set));
  }
  return true;
}

bool GraphSignalManager::AcquireSet(amd::Device* device, int count,
                                    std::vector<void*>& out_set) {
  if (count <= 0) {
    out_set.clear();
    return true;
  }

  {
    std::lock_guard<std::mutex> lock(lock_);
    auto& pool = free_sets_[device];
    if (!pool.empty()) {
      // Hot path: just hand out a ready (already-armed) set, then patch it.
      out_set = std::move(pool.back());
      pool.pop_back();
      return true;
    }
  }

  // Fallback only: more launches in flight than pre-created sets. Create one
  // (armed to 1 by CreateHwEvents); it joins the pool when released.
  return device->CreateHwEvents(count, out_set);
}

void GraphSignalManager::ReleaseSet(amd::Device* device, std::vector<void*>& set) {
  if (set.empty()) {
    return;
  }
  // Re-arm the signals for the next launch. Safe here because this runs from
  // the launch completion callback, so the GPU work that used them is done.
  device->ResetHwEvents(set);
  std::lock_guard<std::mutex> lock(lock_);
  free_sets_[device].push_back(std::move(set));
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
