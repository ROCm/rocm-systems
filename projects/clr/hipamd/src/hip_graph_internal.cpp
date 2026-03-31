/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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
}

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
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Add %s(%p)",
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
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Root node: %s(%p)",
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

    // Process child graph separately, since, there is no connection
    if (node->GetType() == hipGraphNodeTypeGraph) {
      auto child = reinterpret_cast<hip::ChildGraphNode*>(node)->GetChildGraph();
      child->ScheduleNodes();
      max_streams_ = std::max(max_streams_, child->max_streams_);
      //if (child->max_streams_ == 1) {
      reinterpret_cast<hip::ChildGraphNode*>(node)->GraphExec::TopologicalOrder();
      //}
    }
    for (auto edge: node->GetEdges()) {
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
    for (auto dep: entry->GetDependencies()) {
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
  while (!q.empty())
  {
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
  auto curDevId = ihipGetDevice();
  for (hip::GraphNode* entry : vertices_) {
    GraphNode* node = entry->clone();
    node->SetDeviceId(curDevId);
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
  if(!cloneNodes) {
    newGraph->clonedNodes_.clear();
  }
}

// ================================================================================================
Graph* Graph::clone() const {
  Graph* newGraph = new Graph(device_);
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
hipError_t GraphExec::CreateStreams(uint32_t num_streams) {
  parallel_streams_.reserve(num_streams);
  for (uint32_t i = 0; i < num_streams; ++i) {
    auto stream = new hip::Stream(hip::getCurrentDevice(),
                                  hip::Stream::Priority::Normal, hipStreamNonBlocking);
    if (stream == nullptr || !stream->Create()) {
      if (stream != nullptr) {
        hip::Stream::Destroy(stream);
      }
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to create parallel stream!");
      return hipErrorOutOfMemory;
    }
    parallel_streams_.push_back(stream);
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t GraphExec::Init() {
  hipError_t status = hipSuccess;
  // create extra stream to avoid queue collision with the default execution stream
  if (max_streams_ > 1) {
    status = CreateStreams(max_streams_);
  }
  if (status != hipSuccess) {
    return status;
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    if (max_streams_ == 1) {
      // For graph nodes capture AQL packets to dispatch them directly during graph launch.
      status = CaptureAQLPackets();
    }
  }
  instantiateDeviceId_ = hip::getCurrentDevice()->deviceId();
  static_cast<ReferenceCountedObject*>( hip::getCurrentDevice())->retain();
  return status;
}

//! Chunk size to add to kern arg pool
constexpr uint32_t kKernArgChunkSize = 128 * Ki;
// ================================================================================================
void GraphExec::GetKernelArgSizeForGraph(size_t& kernArgSizeForGraph) {
  // GPU packet capture is enabled for kernel nodes. Calculate the kernel
  // arg size required for all graph kernel nodes to allocate
  for (hip::GraphNode* node : topoOrder_) {
    if (node->GraphCaptureEnabled()) {
      kernArgSizeForGraph += node->GetKerArgSize();
    } else if (node->GetType() == hipGraphNodeTypeGraph) {
      auto childNode = reinterpret_cast<hip::ChildGraphNode*>(node);
      // Child graph shares same kernel arg manager
      GraphKernelArgManager* KernelArgManager = GetKernelArgManager();
      KernelArgManager->retain();
      childNode->SetKernelArgManager(KernelArgManager);
      if (childNode->GetChildGraph()->max_streams_ == 1) {
        childNode->GetKernelArgSizeForGraph(kernArgSizeForGraph);
      }
    }
  }
}

// ================================================================================================
hipError_t GraphExec::AllocKernelArgForGraphNode() {
  hipError_t status = hipSuccess;
  for (auto& node : topoOrder_) {
    if (node->GetType() == hipGraphNodeTypeKernel) {
      // Check if graph requires hidden heap and set as part of graphExec param.
      static bool initialized = false;
      if (!initialized && reinterpret_cast<hip::GraphKernelNode*>(node)->HasHiddenHeap()) {
        SetHiddenHeap();
        initialized = true;
      }
    }
    if (node->GraphCaptureEnabled()) {
      status = node->CaptureAndFormPacket(GetKernelArgManager());
    } else if (node->GetType() == hipGraphNodeTypeGraph) {
      auto childNode = reinterpret_cast<hip::ChildGraphNode*>(node);
      if (childNode->GetChildGraph()->max_streams_ == 1) {
        childNode->SetGraphCaptureStatus(true);
        status = childNode->AllocKernelArgForGraphNode();
        if (status != hipSuccess) {
          return status;
        }
      }
    }
  }
  return status;
}

// ================================================================================================
hipError_t GraphExec::CaptureAQLPackets() {
  hipError_t status = hipSuccess;
  size_t kernArgSizeForGraph = 0;
  GetKernelArgSizeForGraph(kernArgSizeForGraph);
  // When we support multi device graph lauch we need to allocate the kenel args on respective
  // device for each kernel Assume graph has nodes of same device allocate kernel args on the device
  // from the first node
  auto device = g_devices[topoOrder_[0]->GetDeviceId()]->devices()[0];
  // Add a larger initial pool to accomodate for any updates to kernel args
  bool bStatus =
      kernArgManager_->AllocGraphKernargPool(kernArgSizeForGraph + kKernArgChunkSize, device);
  if (bStatus != true) {
    return hipErrorMemoryAllocation;
  }

  status = AllocKernelArgForGraphNode();
  if (status != hipSuccess) {
    return status;
  }
  kernArgManager_->ReadBackOrFlush();
  return status;
}

// ================================================================================================
hipError_t GraphExec::UpdateAQLPacket(hip::GraphNode* node) {
  hipError_t status = hipSuccess;
  if (max_streams_ == 1) {
    status = node->CaptureAndFormPacket(kernArgManager_);
  }
  return status;
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
  for (int i = 0; i < topoOrder_.size(); i++) {
    if (topoOrder_[i]->GraphCaptureEnabled()) {
      if (topoOrder_[i]->GetEnabled()) {
        std::vector<uint8_t*>& gpuPackets = topoOrder_[i]->GetAqlPackets();
        for (auto& packet : gpuPackets) {
          hip_stream->vdev()->dispatchAqlPacket(packet, topoOrder_[i]->GetKernelName(), accumulate);
        }
      }
    } else {
      topoOrder_[i]->SetStream(hip_stream);
      status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
      topoOrder_[i]->EnqueueCommands(hip_stream);
    }
  }

  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    accumulate->enqueue();
    accumulate->release();
  }
  return status;
}

// ================================================================================================
void Graph::UpdateStreams(hip::Stream* launch_stream,
                          const std::vector<hip::Stream*>& parallel_streams) {
  // Allocate array for parallel streams, based on the graph scheduling + current stream
  // We create extra stream to avoid collision
  streams_.resize(max_streams_);
  // Current stream is the default in the assignment
  streams_[0] = launch_stream;
  // Assign the streams in the array of all streams
  // Avoid stream that has collision with launch stream
  for (uint32_t i = 1, j = 0; i < streams_.size(); j++) {
    assert(j != parallel_streams.size());
    if (launch_stream->getQueueID() != parallel_streams[j]->getQueueID()) {
      streams_[i++] = parallel_streams[j];
    }
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
          waits_count_++;
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
    node->hw_queue_id_ = node->GetQueue()->getQueueID();
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

// Path Decomposition: assign each node to a unique path starting from roots, printing all such paths, sorted by length
bool Graph::PathDecomposition() {
  const auto& topo_order = GetTopoOrder();

  std::vector< int > stream0, stream1, stream2;
  stream0.reserve(200);
  stream1.reserve(200);
  stream2.reserve(200);
  stream0 = {1,4,6,8,13};
  stream1 = {0,3};

  // 7,9,16,18,25,27,34,36,43,45

  for (int i = 0, j = 15, k = 10, l = 7; i < 40; i++) {
    stream0.push_back(j);
    stream0.push_back(j + 2);
    stream0.push_back(j + 6);
    stream0.push_back(j + 7);
    stream1.push_back(k);
    stream1.push_back(k + 1);

    stream2.push_back(l);
    stream2.push_back(l + 2);
    j += 9; k += 9; l += 9;
  }

  int I = 0;
  for (auto node : topo_order) {
    node->Zid = I;
    if (std::find(stream0.begin(), stream0.end(), node->Zid) != stream0.end()) {
      node->stream_id_ = 0;
    } else if (std::find(stream1.begin(), stream1.end(), node->Zid) != stream1.end()) {
      node->stream_id_ = 1;
    } else if (std::find(stream2.begin(), stream2.end(), node->Zid) != stream2.end()) {
      node->stream_id_ = 2;
    } else {
      node->stream_id_ = 3;
    }
    I++;
  }
  return true;
}

std::unordered_map<Node, int> ComputeCriticalPath(Graph& g) {
    std::unordered_map<Node, int> cp;
    const auto& topo = g.GetTopoOrder();
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Node u = *it;
        int best = 0;
        for (Node v : u->GetEdges()) {
            best = std::max(best, cp[v]);
        }
        cp[u] = 1 + best;
    }
    return cp;
}

std::vector<ScheduledNode> ScheduleGraph(Graph& g,
                                       double alpha = 1.0,   // weight for CP
                                       double beta  = 0.3,   // weight for fanout
                                       double new_stream_penalty = 1.0,
                                       int max_streams = -1) {

  std::vector<ScheduledNode> result;

  auto cp = ComputeCriticalPath(g);

  std::unordered_map<Node, int> indegree;
  for (Node n : g.GetNodes()) {
      indegree[n] = n->GetInDegree();
  }

  // Ready queue with priority
  auto cmp = [&](Node a, Node b) {
      double pa = alpha * cp[a] + beta * a->GetOutDegree();
      double pb = alpha * cp[b] + beta * b->GetOutDegree();
      return pa < pb; // max-heap
  };

  std::priority_queue<Node, std::vector<Node>, decltype(cmp)> ready(cmp);

  for (Node n : g.GetNodes()) {
      if (indegree[n] == 0) {
          ready.push(n);
      }
  }

  std::unordered_map<Node, int> stream;
  int next_stream = 0;

  while (!ready.empty()) {
    Node u = ready.top();
    ready.pop();

    // ---- PICK MAIN PARENT ----
    Node main_parent = nullptr;
    int best_cp = -1;

    for (Node dep : u->GetDependencies()) {
        if (cp[dep] > best_cp) {
            best_cp = cp[dep];
            main_parent = dep;
        }
    }

    // ---- ASSIGN STREAM ----
    if (!main_parent) {
        // root
        stream[u] = next_stream % max_streams;
        next_stream++;
    } else {
        // stream[u] = stream[main_parent];
    }

    result.push_back({u, stream[u]});

    // ---- PROCESS CHILDREN ----
    auto children = u->GetEdges();

    // sort children by cp descending
    std::sort(children.begin(), children.end(), [&](Node a, Node b) {
        return cp[a] > cp[b];
    });

    int alt_stream = (stream[u] + 1) % max_streams;

    for (size_t i = 0; i < children.size(); i++) {
        Node v = children[i];

        indegree[v]--;

        if (indegree[v] == 0) {
            // enforce stream for children:
            if (i == 0) {
                // main continuation
                stream[v] = stream[u];
            } else {
                stream[v] = alt_stream;
                alt_stream = (alt_stream + 1) % max_streams;
            }

            ready.push(v);
        }
    }
  }

  return result;
}

// ================================================================================================
bool Graph::RunNodes(
    int32_t base_stream,
    const std::vector<hip::Stream*>* parallel_streams,
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
      if ((base_stream != i) && (roots_[i] != nullptr)) 
      {
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

  auto start_time = std::chrono::high_resolution_clock::now();

  if (scheduled_nodes_.empty()) {
    scheduled_nodes_ = ScheduleGraph(*this,
      /*alpha*/ 1.0,
      /*beta*/ 0.3,
      /*new_stream_penalty*/ 1000,
      /*max_streams*/ DEBUG_HIP_FORCE_GRAPH_QUEUES);
    // PathDecomposition();
  }

  // Run all commands in the graph
  waits_count_ = 0;
#if 1
  for (auto scheduled_node : scheduled_nodes_) {
    Node node = scheduled_node.node;
    // node->stream_id_ = scheduled_node.stream % DEBUG_HIP_FORCE_GRAPH_QUEUES;
#else
  for (auto node : GetTopoOrder()) {
#endif
    node->launch_id_ = -1;
    if (!RunOneNode(node)) return false;
  }
  XPUT("%p: waits_count_: %d", this, waits_count_);
  
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
    if (end_marker != nullptr) {
      end_marker->enqueue();
      end_marker->release();
    }
    for (auto command : wait_list) {
      command->release();
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
  }  else {
    repeatLaunch_ = true;
  }

  if (max_streams_ == 1 && instantiateDeviceId_ == launch_stream->DeviceId()) {
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
  } else if (max_streams_ == 1 && instantiateDeviceId_ != launch_stream->DeviceId()) {
    for (int i = 0; i < topoOrder_.size(); i++) {
      topoOrder_[i]->SetStream(launch_stream);
      status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
      topoOrder_[i]->EnqueueCommands(launch_stream);
    }
  } else {
    // Update streams for the graph execution
    UpdateStreams(launch_stream, parallel_streams_);
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
  // Current device is stored as part of tls. Save current device to destroy kernelArgs from the
  // callback thread.
  device_ = device;
  if (device->info().largeBar_) {
    graph_kernarg_base = reinterpret_cast<address>(device->deviceLocalAlloc(pool_size));
    device_kernarg_pool_ = true;
  } else {
    graph_kernarg_base = reinterpret_cast<address>(
        device->hostAlloc(pool_size, 0, amd::Device::MemorySegment::kKernArg));
  }

  if (graph_kernarg_base == nullptr) {
    return false;
  }
  kernarg_graph_.push_back(KernelArgPoolGraph(graph_kernarg_base, pool_size));
  return true;
}

address GraphKernelArgManager::AllocKernArg(size_t size, size_t alignment) {
  assert(alignment != 0);
  address result = nullptr;
  result = amd::alignUp(
      kernarg_graph_.back().kernarg_pool_addr_ + kernarg_graph_.back().kernarg_pool_offset_,
      alignment);
  const size_t pool_new_usage = (result + size) - kernarg_graph_.back().kernarg_pool_addr_;
  if (pool_new_usage <= kernarg_graph_.back().kernarg_pool_size_) {
    kernarg_graph_.back().kernarg_pool_offset_ = pool_new_usage;
  } else {
    // If current chunck is full allocate new chunck with same size as current
    bool bStatus = AllocGraphKernargPool(kernarg_graph_.back().kernarg_pool_size_, device_);
    if (bStatus == false) {
      return nullptr;
    } else {
      // Allocte kernel arg memory from new chunck
      return AllocKernArg(size, alignment);
    }
  }
  return result;
}

void GraphKernelArgManager::ReadBackOrFlush() {
  if (device_kernarg_pool_ && device_) {
    auto kernArgImpl = device_->settings().kernel_arg_impl_;

    if (kernArgImpl == KernelArgImpl::DeviceKernelArgsHDP) {
      *device_->info().hdpMemFlushCntl = 1u;
      auto kSentinel = *reinterpret_cast<volatile int*>(device_->info().hdpMemFlushCntl);
    } else if (kernArgImpl == KernelArgImpl::DeviceKernelArgsReadback &&
               kernarg_graph_.back().kernarg_pool_addr_ != 0) {
      address dev_ptr =
          kernarg_graph_.back().kernarg_pool_addr_ + kernarg_graph_.back().kernarg_pool_size_;
      auto kSentinel = *reinterpret_cast<volatile unsigned char*>(dev_ptr - 1);
      _mm_sfence();
      *(dev_ptr - 1) = kSentinel;
      _mm_mfence();
      kSentinel = *reinterpret_cast<volatile unsigned char*>(dev_ptr - 1);
    }
  }
}
}  // namespace hip
