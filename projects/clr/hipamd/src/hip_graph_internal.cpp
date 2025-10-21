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
// root nodes are all vertices with 0 in-degrees
std::vector<Node> Graph::GetRootNodes() const {
  std::vector<Node> roots;
  for (auto entry : vertices_) {
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
      child->ScheduleNodes();
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
void Graph::ScheduleNodes() {
  for (auto node : vertices_) {
    node->stream_id_ = -1;
    node->signal_is_required_ = false;
  }
  memset(&roots_[0], 0, sizeof(Node) * roots_.size());
  max_streams_ = 0;
  // Start processing all nodes in the graph to find async executions.
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

  // Validate input parameters
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

  parallel_streams_[devId].reserve(num_streams);

  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Creating %u parallel streams for device %d",
          num_streams, devId);

  for (uint32_t i = 0; i < num_streams; ++i) {
    auto stream =
        new hip::Stream(g_devices[devId], hip::Stream::Priority::Normal, hipStreamNonBlocking);

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

  if (max_streams_ == 1) {
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
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      // For graph nodes capture AQL packets to dispatch them directly during graph launch.
      status = CaptureAQLPackets();
    }
  } else {
    status = CreateStreams(max_streams_, hip::getCurrentDevice()->deviceId());
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
  for (hip::GraphNode* node : topoOrder_) {
    if (node->GraphCaptureEnabled()) {
      // Accumulate the kernel argument size for each device
      kernArgSizeForGraph[node->dev_id_] += node->GetKerArgSize();
    } else if (node->GetType() == hipGraphNodeTypeGraph) {
      // Handle child graph nodes
      auto childNode = reinterpret_cast<hip::ChildGraphNode*>(node);

      // Child graph shares same kernel arg manager
      GraphKernelArgManager* KernelArgManager = GetKernelArgManager();
      if (KernelArgManager != nullptr) {
        KernelArgManager->retain();
        childNode->SetKernelArgManager(KernelArgManager);
        // Recursively process child graph if it uses single stream
        if (childNode->GetChildGraph()->max_streams_ == 1) {
          childNode->GetKernelArgSizeForGraph(kernArgSizeForGraph);
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
hipError_t GraphExec::CaptureAndFormPacketsForGraph() {
  hipError_t status = hipSuccess;

  // Clear previous capture status and batches
  nodeCaptureStatus_.clear();
  nodeCaptureStatus_.resize(topoOrder_.size(), false);

  // Clear previous batches
  packetBatches_.clear();

  // Process nodes and create batches of consecutive captured nodes
  for (size_t i = 0; i < topoOrder_.size(); ++i) {
    auto& node = topoOrder_[i];

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
      // TODO: Add support for batching for multi-device linear graph
      if (max_streams_dev_.size() == 1) {
        // Single device - use batching optimization
        // Start of a new batch
        PacketBatch newBatch;
        size_t j = i;

        // Collect packets from consecutive captured nodes
        while (j < topoOrder_.size() && topoOrder_[j]->GraphCaptureEnabled()) {
          auto& currentNode = topoOrder_[j];

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
          PacketBatch::NodeRange range;
          range.startIndex = newBatch.dispatchPackets.size();
          range.packetCount = nodePackets.size();
          range.enabled = true;

          // Add to dispatch lists (initially all enabled)
          newBatch.dispatchPackets.insert(newBatch.dispatchPackets.end(), nodePackets.begin(),
                                          nodePackets.end());
          newBatch.dispatchKernelNames.insert(newBatch.dispatchKernelNames.end(),
                                              nodeKernelNames.begin(), nodeKernelNames.end());

          // Store node mapping
          newBatch.nodeRanges.push_back(range);
          newBatch.nodeToRangeIndex[currentNode] = newBatch.nodeRanges.size() - 1;

          // Mark this node as successfully captured
          nodeCaptureStatus_[j] = true;
          ++j;
        }

        // Add the batch if it has packets
        if (!newBatch.dispatchPackets.empty()) {
          packetBatches_.emplace_back(std::move(newBatch));
        }

        // Skip the nodes we just processed, the index will be incremented by the loop
        i = j - 1;
      } else {
        // Multi-device - capture individual packets without batching
        status = node->CaptureAndFormPacket(GetKernelArgManager());
        if (status != hipSuccess) {
          LogError("Individual packet capture failed for multi-device node");
          return status;
        }
      }
    } else if (node->GetType() == hipGraphNodeTypeGraph) {
      auto childNode = reinterpret_cast<hip::ChildGraphNode*>(node);
      if (childNode->GetChildGraph()->max_streams_ == 1) {
        childNode->SetGraphCaptureStatus(true);
        status = childNode->CaptureAndFormPacketsForGraph();
        nodeCaptureStatus_[i] = (status == hipSuccess);
        if (status != hipSuccess) {
          LogWarning("Child graph packet capture failed continuing with other nodes");
          status = hipSuccess;  // Continue processing other nodes
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
  if (max_streams_ != 1 || !node->GraphCaptureEnabled()) {
    return hipSuccess;
  }
  //ToDo: Add batching support for multi-device linear graph
  if (max_streams_dev_.size() == 1) {
    // Find which batch contains this node and update it
    for (auto& batch : packetBatches_) {
      auto it = batch.nodeToRangeIndex.find(node);
      if (it != batch.nodeToRangeIndex.end()) {
        // Found the batch containing this node - update packets
        PacketBatch::NodeRange& range = batch.nodeRanges[it->second];

        // Capture new packets for this node
        std::vector<uint8_t*> newPackets;
        std::vector<std::string> newKernelNames;
        hipError_t status =
            node->CaptureAndFormPacket(kernArgManager_, &newPackets, &newKernelNames);
        if (status != hipSuccess) {
          return status;
        }
        // Update dispatch packets (always update regardless of enabled state)
        // The enabled/disabled check happens during dispatch, not here
        for (size_t i = 0; i < range.packetCount && i < newPackets.size(); ++i) {
          size_t packetIndex = range.startIndex + i;
          batch.dispatchPackets[packetIndex] = newPackets[i];
          batch.dispatchKernelNames[packetIndex] = newKernelNames[i];
        }
        return hipSuccess;
      }
    }
  } else {
    return node->CaptureAndFormPacket(kernArgManager_);
  }
  return hipSuccess; // Node not in any batch
}

// ================================================================================================
hipError_t GraphExec::UpdatePacketBatchesForNodeEnableDisable(hip::GraphNode* node, bool isEnabled) {
  if (max_streams_ != 1 && max_streams_dev_.size() == 1 && !node->GraphCaptureEnabled()) {
    // Only handle single stream and single device case with captured nodes
    return hipSuccess;
  }
  // Find which batch contains this node and update its enabled state
  for (auto& batch : packetBatches_) {
    auto it = batch.nodeToRangeIndex.find(node);
    if (it != batch.nodeToRangeIndex.end()) {
      // Found the batch containing this node - update enabled state
      batch.setEnabled(node, isEnabled);
      return hipSuccess;
    }
  }
  return hipSuccess; // Node not in any batch
}

// ================================================================================================

void GraphExec::DecrementRefCount(cl_event event, cl_int command_exec_status, void* user_data) {
  GraphExec* graphExec = reinterpret_cast<GraphExec*>(user_data);
  graphExec->release();
}

// ================================================================================================

hipError_t GraphExec::EnqueueGraphWithSingleList(hip::Stream* hip_stream) {
  // Accumulate command tracks all the AQL packet batch that we submit to the HW. For now
  // we track only kernel nodes.
  amd::AccumulateCommand* accumulate = nullptr;
  hipError_t status = hipSuccess;
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    accumulate = new amd::AccumulateCommand(*hip_stream, {}, nullptr);
  }

  size_t batchIndex = 0;

  // Process nodes in topological order with mixed execution strategy
  for (size_t i = 0; i < topoOrder_.size(); ++i) {
    auto& node = topoOrder_[i];

    if (!node->GraphCaptureEnabled()) {
      // Node doesn't support capture - execute individually if enabled
      if (node->GetEnabled() != 0) {
        node->SetStream(hip_stream);
        status = node->CreateCommand(node->GetQueue());
        node->EnqueueCommands(hip_stream);
      }
    } else if (i < nodeCaptureStatus_.size() && nodeCaptureStatus_[i]) {
      // Node was successfully captured - dispatch the batch with enabled nodes only
      if (batchIndex < packetBatches_.size()) {
        const auto& batch = packetBatches_[batchIndex];
        // O(1) check: if no disabled nodes, dispatch entire batch directly
        // This avoids creating new vectors when all nodes are enabled (common case)
        if (batch.disabledNodeCount == 0) {
          // Fast path: all nodes enabled, dispatch entire batch
          bool batchStatus = hip_stream->vdev()->dispatchAqlPacketBatch(
              batch.dispatchPackets, batch.dispatchKernelNames, accumulate);
          if (!batchStatus) {
            status = hipErrorUnknown;
            accumulate->release();
            return status;
          }
        } else {
          // Slow path: some nodes disabled, create filtered vectors
          std::vector<uint8_t*> enabledPackets;
          std::vector<std::string> enabledKernelNames;
          for (const auto& range : batch.nodeRanges) {
            if (range.enabled) {
              // Add packets for this enabled node
              for (size_t j = 0; j < range.packetCount; ++j) {
                size_t packetIndex = range.startIndex + j;
                enabledPackets.push_back(batch.dispatchPackets[packetIndex]);
                enabledKernelNames.push_back(batch.dispatchKernelNames[packetIndex]);
              }
            }
          }
          // Only dispatch if there are enabled packets
          if (!enabledPackets.empty()) {
            bool batchStatus = hip_stream->vdev()->dispatchAqlPacketBatch(
                enabledPackets, enabledKernelNames, accumulate);
            if (!batchStatus) {
              status = hipErrorUnknown;
              accumulate->release();
              return status;
            }
          }
        }

        // Skip all consecutive captured nodes that belong to this batch
        i += packetBatches_[batchIndex].nodeRanges.size() - 1;  // -1 because loop will increment

        ++batchIndex;
      }
    }
  }

  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    accumulate->enqueue();
    accumulate->release();
  }
  return status;
}
// ================================================================================================
hipError_t GraphExec::EnqueueMultiDeviceLinearGraph(hip::Stream* launch_stream) {
  // Accumulate command tracks all the AQL packet batch that we submit to the HW. For now we track
  // only kernel nodes.
  amd::AccumulateCommand* accumulate = nullptr;
  hipError_t status = hipSuccess;
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    accumulate = new amd::AccumulateCommand(*launch_stream, {}, nullptr);
  }

  auto createMarkerAndWait = [](hip::Stream* fromStream, hip::Stream* toStream) {
    amd::Command::EventWaitList wait_list;
    auto marker = new amd::Marker(*fromStream, true, wait_list);
    marker->enqueue();
    marker->release();
    wait_list.push_back(marker);
    auto wait_marker = new amd::Marker(*toStream, true, wait_list);
    wait_marker->enqueue();
    wait_marker->release();
  };

  hip::Stream* prevStream = launch_stream;
  size_t batchIndex = 0;

  for (size_t i = 0; i < topoOrder_.size(); ++i) {
    auto& node = topoOrder_[i];
    hip::Stream* currStream = parallel_streams_[node->dev_id_][0];

    // Insert synchronization marker if switching devices
    if (prevStream->DeviceId() != currStream->DeviceId()) {
      createMarkerAndWait(prevStream, currStream);
    }
    // ToDo : Add batching for multi device graph launch
    if (topoOrder_[i]->GraphCaptureEnabled()) {
      if (topoOrder_[i]->GetEnabled()) {
        std::vector<uint8_t*>& gpuPackets = topoOrder_[i]->GetAqlPackets();
        std::vector<std::string> kernelNames;
        for (auto& packet : gpuPackets) {
          kernelNames.push_back(topoOrder_[i]->GetKernelName());
        }
        currStream->vdev()->dispatchAqlPacketBatch(gpuPackets, kernelNames, accumulate);
      }
    } else {
      topoOrder_[i]->SetStream(currStream);
      status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
      topoOrder_[i]->EnqueueCommands(currStream);
    }
    prevStream = currStream;
  }

  // Synchronize back to launch stream if we ended on a different device
  if (prevStream != launch_stream) {
    createMarkerAndWait(prevStream, launch_stream);
  }

  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    accumulate->enqueue();
    accumulate->release();
  }

  return status;
}

// ================================================================================================
void GraphExec::UpdateStreams(hip::Stream* launch_stream) {
  int devId = launch_stream->vdev()->device().index();
  if (parallel_streams_.find(devId) == parallel_streams_.end()) {
    LogPrintfError("UpdateStreams failed for device id:%d", devId);
    return;
  }
  auto parallel_streams = parallel_streams_[devId];
  // Current stream is the default in the assignment
  streams_.push_back(launch_stream);
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
  if (flags_ & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    if (!topoOrder_.empty()) {
      topoOrder_[0]->GetParentGraph()->FreeAllMemory(launch_stream);
      topoOrder_[0]->GetParentGraph()->memalloc_nodes_ = 0;
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
    if (!topoOrder_.empty() && topoOrder_[0]->GetParentGraph()->GetMemAllocNodeCount() > 0) {
      return hipErrorInvalidValue;
    }
  } else {
    repeatLaunch_ = true;
  }
  ClPrint(amd::LOG_DEBUG, amd::LOG_CODE,
          "GraphExec::Run max_streams: %d, "
          "on device: %d, total number of nodes: %d",
          max_streams_, launch_stream->DeviceId(), topoOrder_.size());

  if (max_streams_ == 1 && max_streams_dev_.size() == 1 &&
      max_streams_dev_.begin()->first == launch_stream->DeviceId()) {
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      // If the graph has kernels that does device side allocation,  during packet capture, heap is
      // allocated because heap pointer has to be added to the AQL packet, and initialized during
      // graph launch.
      static bool initialized = false;
      if (!initialized && HasHiddenHeap()) {
        launch_stream->vdev()->HiddenHeapInit();
        initialized = true;
      }
    }
    status = EnqueueGraphWithSingleList(launch_stream);
  } else if (max_streams_ == 1 && max_streams_dev_.size() > 1) {
    status = EnqueueMultiDeviceLinearGraph(launch_stream);
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
