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

#include "top.hpp"
#include "hip_graph_internal.hpp"
#include "platform/command.hpp"
#include "hip_conversions.hpp"
#include "hip_platform.hpp"
#include "hip_event.hpp"
#include "hip_mempool_impl.hpp"

namespace hip {

std::vector<hip::Stream*> g_captureStreams;
// StreamCaptureGlobalList lock
amd::Monitor g_captureStreamsLock{};
// StreamCaptureset lock
amd::Monitor g_streamSetLock{};
std::unordered_set<hip::Stream*> g_allCapturingStreams;
hipError_t ihipGraphDebugDotPrint(hip::Graph* graph, const char* path, unsigned int flags);
hipError_t ihipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                               size_t numDependencies, unsigned int flags);

inline hipError_t ihipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  if (graphExec == nullptr) {
    return hipErrorInvalidValue;
  }
  getStreamPerThread(stream);
  if (!hip::GraphExec::isGraphExecValid(reinterpret_cast<hip::GraphExec*>(graphExec))) {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

inline hipError_t ihipGraphAddNode(hip::GraphNode* graphNode, hip::Graph* graph,
                                   hip::GraphNode* const* pDependencies, size_t numDependencies,
                                   bool capture, int devId) {
  graph->AddNode(graphNode);
  std::unordered_set<hip::GraphNode*> DuplicateDep;
  for (size_t i = 0; i < numDependencies; i++) {
    if ((!hip::GraphNode::isNodeValid(pDependencies[i])) ||
        (graph != pDependencies[i]->GetParentGraph())) {
      return hipErrorInvalidValue;
    }
    if (DuplicateDep.find(pDependencies[i]) != DuplicateDep.end()) {
      return hipErrorInvalidValue;
    }
    DuplicateDep.insert(pDependencies[i]);
    pDependencies[i]->AddEdgeDep(graphNode);
  }
  if (capture == false) {
    {
      amd::ScopedLock lock(g_streamSetLock);
      for (auto stream : g_allCapturingStreams) {
        if (stream->GetCaptureGraph() == graph) {
          graph->AddManualNodeDuringCapture(graphNode);
          break;
        }
      }
    }
  }
  if (devId != 0) {
    graphNode->SetDeviceId(devId);
  }
  return hipSuccess;
}

hipError_t ihipGraphAddKernelNode(hip::GraphNode** pGraphNode, hip::Graph* graph,
                                  hip::GraphNode* const* pDependencies, size_t numDependencies,
                                  const hipKernelNodeParams* pNodeParams,
                                  const ihipExtKernelEvents* pNodeEvents = nullptr,
                                  bool capture = true, int coopKernel = 0, int devId = 0,
                                  int globalWorkSizeX_remainder = 0,
                                  int globalWorkSizeY_remainder = 0,
                                  int globalWorkSizeZ_remainder = 0) {
  if (!hip::Graph::isGraphValid(graph)) {
    return hipErrorInvalidValue;
  }

  hipFunction_t func = hip::GraphKernelNode::getFunc(*pNodeParams, ihipGetDevice());
  if (!func) {
    return hipErrorInvalidDeviceFunction;
  }

  amd::HIPLaunchParams launch_params(pNodeParams->gridDim.x, pNodeParams->gridDim.y,
                                     pNodeParams->gridDim.z, pNodeParams->blockDim.x,
                                     pNodeParams->blockDim.y, pNodeParams->blockDim.z,
                                     pNodeParams->sharedMemBytes, globalWorkSizeX_remainder,
                                     globalWorkSizeY_remainder, globalWorkSizeZ_remainder);
  if (!launch_params.IsValidConfig()) {
    return hipErrorInvalidConfiguration;
  }
  hipError_t status = ihipLaunchKernel_validate(func, launch_params, pNodeParams->kernelParams,
                                                pNodeParams->extra, ihipGetDevice(), 0);
  if (hipSuccess != status) {
    return status;
  }

  size_t globalWorkSizeX = static_cast<size_t>(pNodeParams->gridDim.x) * pNodeParams->blockDim.x +
      globalWorkSizeX_remainder;
  size_t globalWorkSizeY = static_cast<size_t>(pNodeParams->gridDim.y) * pNodeParams->blockDim.y +
      globalWorkSizeY_remainder;
  size_t globalWorkSizeZ = static_cast<size_t>(pNodeParams->gridDim.z) * pNodeParams->blockDim.z +
      globalWorkSizeZ_remainder;
  if (globalWorkSizeX > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeY > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeZ > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidConfiguration;
  }

  *pGraphNode =
      new hip::GraphKernelNode(pNodeParams, pNodeEvents, coopKernel, globalWorkSizeX_remainder,
                               globalWorkSizeY_remainder, globalWorkSizeZ_remainder);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies, capture, devId);
  return status;
}

hipError_t ihipGraphAddMemcpyNode(hip::GraphNode** pGraphNode, hip::Graph* graph,
                                  hip::GraphNode* const* pDependencies, size_t numDependencies,
                                  const hipMemcpy3DParms* pCopyParams, bool capture = true,
                                  int devId = 0) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pCopyParams == nullptr) {
    return hipErrorInvalidValue;
  }
  hipError_t status = ihipMemcpy3D_validate(pCopyParams);
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hip::GraphMemcpyNode(pCopyParams);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies, capture);
  return status;
}

hipError_t ihipDrvGraphAddMemcpyNode(hip::GraphNode** pGraphNode, hip::Graph* graph,
                                     hip::GraphNode* const* pDependencies, size_t numDependencies,
                                     const HIP_MEMCPY3D* pCopyParams, hipCtx_t ctx,
                                     bool capture = true, int devId = 0) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pCopyParams == nullptr) {
    return hipErrorInvalidValue;
  }
  hipError_t status = ihipDrvMemcpy3D_validate(pCopyParams);
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hip::GraphDrvMemcpyNode(pCopyParams);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies, capture);
  return status;
}

hipError_t ihipGraphAddMemcpyNode1D(hip::GraphNode** pGraphNode, hip::Graph* graph,
                                    hip::GraphNode* const* pDependencies, size_t numDependencies,
                                    void* dst, const void* src, size_t count, hipMemcpyKind kind,
                                    bool capture = true, int devId = 0) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || count == 0) {
    return hipErrorInvalidValue;
  }
  hipError_t status = hip::GraphMemcpyNode1D::ValidateParams(dst, src, count, kind);
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hip::GraphMemcpyNode1D(dst, src, count, kind);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies, capture);
  return status;
}

hipError_t ihipGraphAddMemsetNode(hip::GraphNode** pGraphNode, hip::Graph* graph,
                                  hip::GraphNode* const* pDependencies, size_t numDependencies,
                                  const hipMemsetParams* pMemsetParams, bool capture = true,
                                  size_t depth = 1, size_t arrWidth = 1, size_t arrHeight = 1,
                                  int devId = 0) {
  if (pGraphNode == nullptr || graph == nullptr || pMemsetParams == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pMemsetParams->height == 0) {
    return hipErrorInvalidValue;
  }
  // The element size must be 1, 2, or 4 bytes
  if (pMemsetParams->elementSize != sizeof(int8_t) &&
      pMemsetParams->elementSize != sizeof(int16_t) &&
      pMemsetParams->elementSize != sizeof(int32_t)) {
    return hipErrorInvalidValue;
  }
  hipError_t status;
  status = ihipGraphMemsetParams_validate(pMemsetParams);
  if (status != hipSuccess) {
    return status;
  }
  if (depth == 0) {
    return hipErrorInvalidValue;
  }
  if (pMemsetParams->height == 1) {
    status =
        ihipMemset_validate(pMemsetParams->dst, pMemsetParams->value, pMemsetParams->elementSize,
                            pMemsetParams->width * pMemsetParams->elementSize);
  } else {
    if (pMemsetParams->pitch < (pMemsetParams->width * pMemsetParams->elementSize)) {
      return hipErrorInvalidValue;
    }
    auto sizeBytes =
        pMemsetParams->width * pMemsetParams->height * depth * pMemsetParams->elementSize;
    status = ihipMemset3D_validate(
        {pMemsetParams->dst, pMemsetParams->pitch, pMemsetParams->width, pMemsetParams->height},
        pMemsetParams->value, {pMemsetParams->width, pMemsetParams->height, depth}, sizeBytes);
  }
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hip::GraphMemsetNode(pMemsetParams, depth, arrWidth, arrHeight);
  status = ihipGraphAddNode(*pGraphNode, graph, pDependencies, numDependencies, capture, devId);
  return status;
}

hipError_t capturehipLaunchKernel(hipStream_t& stream, const void*& hostFunction, dim3& gridDim,
                                  dim3& blockDim, void**& args, size_t& sharedMemBytes) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node LaunchKernel on stream : %p", stream);

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipKernelNodeParams nodeParams;
  nodeParams.func = const_cast<void*>(hostFunction);
  nodeParams.blockDim = blockDim;
  nodeParams.extra = nullptr;
  nodeParams.gridDim = gridDim;
  nodeParams.kernelParams = args;
  nodeParams.sharedMemBytes = sharedMemBytes;

  hip::GraphNode* pGraphNode;
  hipError_t status = ihipGraphAddKernelNode(
      &pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
      s->GetLastCapturedNodes().size(), &nodeParams, nullptr, true, 0, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t ihipExtLaunchKernel(hipStream_t stream, hipFunction_t f, uint32_t globalWorkSizeX,
                               uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                               uint32_t globalWorkSizeX_remainder,
                               uint32_t globalWorkSizeY_remainder,
                               uint32_t globalWorkSizeZ_remainder, uint32_t localWorkSizeX,
                               uint32_t localWorkSizeY, uint32_t localWorkSizeZ,
                               size_t sharedMemBytes, void** kernelParams, void** extra,
                               hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags,
                               bool capture = true) {
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);

  hip::GraphNode* pGraphNode;
  hipError_t status = hipSuccess;

  // Consider start event as an explicit node because it is recorded as an event
  // For the purpose of optimization, the stop event isn't a separate node as we bind it later
  // to the KernelLaunch command. Pass only the stopEvent further.
  ihipExtKernelEvents nodeEvents;
  nodeEvents.startEvent_ = nullptr;
  nodeEvents.stopEvent_ = stopEvent;

  if (startEvent != nullptr) {
    pGraphNode = new hip::GraphEventRecordNode(startEvent);
    status = ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                              s->GetLastCapturedNodes().size(), capture);
    if (status != hipSuccess) {
      return status;
    }
    s->SetLastCapturedNode(pGraphNode);
  }

  hipKernelNodeParams nodeParams;
  nodeParams.func = f;
  nodeParams.blockDim = dim3(localWorkSizeX, localWorkSizeY, localWorkSizeZ);
  nodeParams.extra = extra;
  nodeParams.gridDim = dim3(globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ);
  nodeParams.kernelParams = kernelParams;
  nodeParams.sharedMemBytes = sharedMemBytes;

  status = ihipGraphAddKernelNode(
      &pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
      s->GetLastCapturedNodes().size(), &nodeParams, &nodeEvents, true, 0, s->DeviceId(),
      globalWorkSizeX_remainder, globalWorkSizeY_remainder, globalWorkSizeZ_remainder);

  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);

  return hipSuccess;
}

hipError_t capturehipExtModuleLaunchKernel(hipStream_t& stream, hipFunction_t& f,
                                           uint32_t& globalWorkSizeX, uint32_t& globalWorkSizeY,
                                           uint32_t& globalWorkSizeZ, uint32_t& localWorkSizeX,
                                           uint32_t& localWorkSizeY, uint32_t& localWorkSizeZ,
                                           size_t& sharedMemBytes, void**& kernelParams,
                                           void**& extra, hipEvent_t& startEvent,
                                           hipEvent_t& stopEvent, uint32_t& flags) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node ExtModuleLaunchKernel on stream : %p", stream);
  return ihipExtLaunchKernel(stream, f, globalWorkSizeX / localWorkSizeX,
                             globalWorkSizeY / localWorkSizeY, globalWorkSizeZ / localWorkSizeZ,
                             globalWorkSizeX % localWorkSizeX, globalWorkSizeY % localWorkSizeY,
                             globalWorkSizeZ % localWorkSizeZ, localWorkSizeX, localWorkSizeY,
                             localWorkSizeZ, sharedMemBytes, kernelParams, extra, startEvent,
                             stopEvent, flags);
}

hipError_t capturehipExtLaunchKernel(hipStream_t& stream, const void*& hostFunction, dim3& gridDim,
                                     dim3& blockDim, void**& args, size_t& sharedMemBytes,
                                     hipEvent_t& startEvent, hipEvent_t& stopEvent, int& flags) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node ExtLaunchKernel on stream : %p", stream);
  return ihipExtLaunchKernel(
      stream, reinterpret_cast<hipFunction_t>(const_cast<void*>(hostFunction)), gridDim.x,
      gridDim.y, gridDim.z, 0, 0, 0, blockDim.x, blockDim.y, blockDim.z, sharedMemBytes, args,
      nullptr, startEvent, stopEvent, flags);
}

hipError_t capturehipModuleLaunchKernel(hipStream_t& stream, hipFunction_t& f, uint32_t& gridDimX,
                                        uint32_t& gridDimY, uint32_t& gridDimZ, uint32_t& blockDimX,
                                        uint32_t& blockDimY, uint32_t& blockDimZ,
                                        uint32_t& sharedMemBytes, void**& kernelParams,
                                        void**& extra) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node ModuleLaunchKernel on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipKernelNodeParams nodeParams;
  nodeParams.func = f;
  nodeParams.blockDim = {blockDimX, blockDimY, blockDimZ};
  nodeParams.extra = extra;
  nodeParams.gridDim = {gridDimX, gridDimY, gridDimZ};
  nodeParams.kernelParams = kernelParams;
  nodeParams.sharedMemBytes = sharedMemBytes;

  hip::GraphNode* pGraphNode;
  hipError_t status = ihipGraphAddKernelNode(
      &pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
      s->GetLastCapturedNodes().size(), &nodeParams, nullptr, true, 0, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipModuleLaunchCooperativeKernel(hipStream_t& stream, hipFunction_t& f,
                                                   uint32_t& gridDimX, uint32_t& gridDimY,
                                                   uint32_t& gridDimZ, uint32_t& blockDimX,
                                                   uint32_t& blockDimY, uint32_t& blockDimZ,
                                                   uint32_t& sharedMemBytes, void**& kernelParams) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node ModuleLaunchCooperativeKernel on stream : %p", stream);

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipKernelNodeParams nodeParams;
  nodeParams.func = f;
  nodeParams.blockDim = {blockDimX, blockDimY, blockDimZ};
  nodeParams.gridDim = {gridDimX, gridDimY, gridDimZ};
  nodeParams.kernelParams = kernelParams;
  nodeParams.sharedMemBytes = sharedMemBytes;
  nodeParams.extra = nullptr;

  hip::GraphNode* pGraphNode;
  hipError_t status =
      ihipGraphAddKernelNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &nodeParams, nullptr, true,
                             amd::NDRangeKernelCommand::CooperativeGroups, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);

  return hipSuccess;
}

hipError_t capturehipLaunchByPtr(hipStream_t& stream, hipFunction_t func, dim3 blockDim,
                                 dim3 gridDim, unsigned int sharedMemBytes, void** extra) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node LaunchByPtr on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  hipKernelNodeParams nodeParams;
  nodeParams.func = func;
  nodeParams.blockDim = blockDim;
  nodeParams.gridDim = gridDim;
  nodeParams.sharedMemBytes = sharedMemBytes;
  nodeParams.extra = extra;
  nodeParams.kernelParams = nullptr;

  hip::GraphNode* pGraphNode;
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipError_t status = ihipGraphAddKernelNode(
      &pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
      s->GetLastCapturedNodes().size(), &nodeParams, nullptr, true, 0, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);

  return hipSuccess;
}

hipError_t capturehipLaunchCooperativeKernel(hipStream_t& stream, const void*& f, dim3& gridDim,
                                             dim3& blockDim, void**& kernelParams,
                                             uint32_t& sharedMemBytes) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node LaunchCooperativeKernel on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipKernelNodeParams nodeParams;
  nodeParams.func = const_cast<void*>(f);
  nodeParams.blockDim = blockDim;
  nodeParams.gridDim = gridDim;
  nodeParams.kernelParams = kernelParams;
  nodeParams.sharedMemBytes = sharedMemBytes;
  nodeParams.extra = nullptr;

  hip::GraphNode* pGraphNode;
  hipError_t status =
      ihipGraphAddKernelNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &nodeParams, nullptr, true,
                             amd::NDRangeKernelCommand::CooperativeGroups, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);

  return hipSuccess;
}

hipError_t capturehipMemcpy3DAsync(hipStream_t& stream, const hipMemcpy3DParms*& p) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memcpy3D on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  // Skip zero-sized copies
  if (p->extent.width == 0 || p->extent.height == 0 || p->extent.depth == 0) {
    return hipSuccess;
  }

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), p, true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy2DAsync(hipStream_t& stream, void*& dst, size_t& dpitch,
                                   const void*& src, size_t& spitch, size_t& width, size_t& height,
                                   hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memcpy2D on stream : %p", stream);
  if (dst == nullptr || src == nullptr) {
    return hipErrorInvalidValue;
  }

  // Skip zero-sized copies
  if (width == 0 || height == 0) {
    return hipSuccess;
  }

  if ((width > dpitch) || (width > spitch)) {
    return hipErrorInvalidPitchValue;
  }

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;

  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.kind = kind;
  p.srcPtr.ptr = const_cast<void*>(src);
  p.srcPtr.pitch = spitch;
  p.srcArray = nullptr;  // Ignored.

  p.dstPtr.ptr = const_cast<void*>(dst);
  p.dstPtr.pitch = dpitch;
  p.dstArray = nullptr;  // Ignored.

  p.extent = {width, height, 1};

  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p, true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy2DFromArrayAsync(hipStream_t& stream, void*& dst, size_t& dpitch,
                                            hipArray_const_t& src, size_t& wOffsetSrc,
                                            size_t& hOffsetSrc, size_t& width, size_t& height,
                                            hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memcpy2DFromArray on stream : %p", stream);

  // Skip zero-sized copies
  if (width == 0 || height == 0) {
    return hipSuccess;
  }

  HIP_RETURN_ONFAIL(hipMemcpy2DValidateArray(src, wOffsetSrc, hOffsetSrc, width, height));
  HIP_RETURN_ONFAIL(hipMemcpy2DValidateBuffer(dst, dpitch, width));

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.srcPos = {wOffsetSrc, hOffsetSrc, 0};
  p.kind = kind;
  p.srcPtr.ptr = nullptr;
  p.srcArray = const_cast<hipArray*>(src);  // Ignored.

  p.kind = kind;
  p.dstPtr.ptr = dst;
  p.dstArray = nullptr;  // Ignored.
  p.dstPtr.pitch = dpitch;
  p.extent = {width / hip::getElementSize(p.srcArray), height, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p, true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy2DToArrayAsync(hipStream_t& stream, hipArray_t& dst, size_t& wOffset,
                                          size_t& hOffset, const void*& src, size_t& spitch,
                                          size_t& width, size_t& height, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memcpy2DFromArray on stream : %p", stream);

  // Skip zero-sized copies
  if (width == 0 || height == 0) {
    return hipSuccess;
  }

  HIP_RETURN_ONFAIL(hipMemcpy2DValidateArray(dst, wOffset, hOffset, width, height));
  HIP_RETURN_ONFAIL(hipMemcpy2DValidateBuffer(src, spitch, width));

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.dstPos = {wOffset, hOffset, 0};
  p.kind = kind;
  p.dstPtr.ptr = nullptr;
  p.dstArray = dst;  // Ignored.

  p.kind = kind;
  p.srcPtr.ptr = const_cast<void*>(src);
  p.srcArray = nullptr;  // Ignored.
  p.srcPtr.pitch = spitch;
  p.extent = {width / hip::getElementSize(p.dstArray), height, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p, true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyParam2DAsync(hipStream_t& stream, const hip_Memcpy2D*& pCopy) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MemcpyParam2D on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  if ((pCopy->srcDevice == nullptr && pCopy->srcMemoryType == hipMemoryTypeDevice) ||
      (pCopy->dstDevice == nullptr && pCopy->dstMemoryType == hipMemoryTypeDevice) ||
      (pCopy->srcHost == nullptr && pCopy->srcMemoryType == hipMemoryTypeHost) ||
      (pCopy->dstHost == nullptr && pCopy->dstMemoryType == hipMemoryTypeHost) ||
      (pCopy->srcArray == nullptr && pCopy->srcMemoryType == hipMemoryTypeArray) ||
      (pCopy->dstArray == nullptr && pCopy->dstMemoryType == hipMemoryTypeArray)) {
    return hipErrorInvalidValue;
  }

  /// Skip zero-sized copies
  if (pCopy->WidthInBytes == 0 || pCopy->Height == 0) {
    return hipSuccess;
  }

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.srcArray = pCopy->srcArray;
  p.srcPos = {pCopy->srcXInBytes, pCopy->srcY, 0};
  p.srcPtr.pitch = pCopy->srcPitch;
  if (pCopy->srcDevice != nullptr) {
    p.srcPtr.ptr = pCopy->srcDevice;
  }
  if (pCopy->srcHost != nullptr) {
    p.srcPtr.ptr = const_cast<void*>(pCopy->srcHost);
  }
  p.dstArray = pCopy->dstArray;
  p.dstPos = {pCopy->dstXInBytes, pCopy->dstY, 0};
  p.dstPtr.pitch = pCopy->dstPitch;
  if (pCopy->dstDevice != nullptr) {
    p.dstPtr.ptr = pCopy->dstDevice;
  }
  if (pCopy->dstHost != nullptr) {
    p.dstPtr.ptr = const_cast<void*>(pCopy->dstHost);
  }


  // If array is participating in the copy, the extent is defined in terms of that array's elements.
  // If no array is participating in the copy then the extents are defined in elements of unsigned
  // char.
  p.extent = {pCopy->WidthInBytes, pCopy->Height, 1};
  if (pCopy->srcArray != nullptr) {
    p.extent.width /= getElementSize(pCopy->srcArray);
  } else if (pCopy->dstArray != nullptr) {
    p.extent.width /= getElementSize(pCopy->dstArray);
  }

  if (pCopy->srcMemoryType == hipMemoryTypeHost && pCopy->dstMemoryType == hipMemoryTypeHost) {
    p.kind = hipMemcpyHostToHost;
  } else if (pCopy->srcMemoryType == hipMemoryTypeHost) {
    p.kind = hipMemcpyHostToDevice;
  } else if (pCopy->dstMemoryType == hipMemoryTypeHost) {
    p.kind = hipMemcpyDeviceToHost;
  } else {
    p.kind = hipMemcpyDeviceToDevice;
  }
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p, true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyAtoHAsync(hipStream_t& stream, void*& dstHost, hipArray_t& srcArray,
                                     size_t& srcOffset, size_t& ByteCount) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MemcpyParam2D on stream : %p", stream);
  if (srcArray == nullptr || dstHost == nullptr) {
    return hipErrorInvalidValue;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.srcArray = srcArray;
  p.srcPos = {srcOffset, 0, 0};
  p.dstPtr.ptr = dstHost;
  p.extent = {ByteCount / hip::getElementSize(p.srcArray), 1, 1};
  p.kind = hipMemcpyDeviceToHost;
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p, true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyHtoAAsync(hipStream_t& stream, hipArray_t& dstArray, size_t& dstOffset,
                                     const void*& srcHost, size_t& ByteCount) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MemcpyParam2D on stream : %p", stream);
  if (dstArray == nullptr || srcHost == nullptr) {
    return hipErrorInvalidValue;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipMemcpy3DParms p = {};
  memset(&p, 0, sizeof(p));
  p.dstArray = dstArray;
  p.dstPos = {dstOffset, 0, 0};
  p.srcPtr.ptr = const_cast<void*>(srcHost);
  p.extent = {ByteCount / hip::getElementSize(p.dstArray), 1, 1};
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &p, true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy(hipStream_t stream, void* dst, const void* src, size_t sizeBytes,
                            hipMemcpyKind kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memcpy on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  std::vector<hip::GraphNode*> pDependencies = s->GetLastCapturedNodes();
  size_t numDependencies = s->GetLastCapturedNodes().size();
  hip::Graph* graph = s->GetCaptureGraph();
  hip::GraphNode* node = new hip::GraphMemcpyNode1D(dst, src, sizeBytes, kind);
  hipError_t status = ihipGraphAddMemcpyNode1D(&node, graph, pDependencies.data(), numDependencies,
                                               dst, src, sizeBytes, kind, true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(node);
  return hipSuccess;
}

hipError_t capturehipMemcpyAsync(hipStream_t& stream, void*& dst, const void*& src,
                                 size_t& sizeBytes, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memcpy1D on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  return capturehipMemcpy(stream, dst, src, sizeBytes, kind);
}

hipError_t capturehipMemcpyHtoDAsync(hipStream_t& stream, hipDeviceptr_t& dstDevice,
                                     const void*& srcHost, size_t& ByteCount,
                                     hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MemcpyHtoD on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  return capturehipMemcpy(stream, dstDevice, srcHost, ByteCount, kind);
}

hipError_t capturehipMemcpyDtoDAsync(hipStream_t& stream, hipDeviceptr_t& dstDevice,
                                     hipDeviceptr_t& srcDevice, size_t& ByteCount,
                                     hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MemcpyDtoD on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  return capturehipMemcpy(stream, dstDevice, srcDevice, ByteCount, kind);
}

hipError_t capturehipMemcpyDtoHAsync(hipStream_t& stream, void*& dstHost,
                                     hipDeviceptr_t& srcDevice, size_t& ByteCount,
                                     hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MemcpyDtoH on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  return capturehipMemcpy(stream, dstHost, srcDevice, ByteCount, kind);
}

hipError_t capturehipMemcpyFromSymbolAsync(hipStream_t& stream, void*& dst, const void*& symbol,
                                           size_t& sizeBytes, size_t& offset,
                                           hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MemcpyFromSymbolNode on stream : %p", stream);

  if (kind != hipMemcpyDeviceToHost && kind != hipMemcpyDeviceToDevice &&
      kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }

  if (dst == nullptr) {
    return hipErrorInvalidValue;
  }

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, sizeBytes, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode =
      new hip::GraphMemcpyNodeFromSymbol(dst, symbol, sizeBytes, offset, kind);
  status = ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                            s->GetLastCapturedNodes().size(), true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyToSymbolAsync(hipStream_t& stream, const void*& symbol, const void*& src,
                                         size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MemcpyToSymbolNode on stream : %p", stream);

  if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToDevice &&
      kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }

  if (src == nullptr) {
    return hipErrorInvalidValue;
  }

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;
  hipError_t status = ihipMemcpySymbol_validate(symbol, sizeBytes, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode =
      new hip::GraphMemcpyNodeToSymbol(symbol, src, sizeBytes, offset, kind);
  status = ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                            s->GetLastCapturedNodes().size(), true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemsetAsync(hipStream_t& stream, void*& dst, int& value, size_t& valueSize,
                                 size_t& sizeBytes) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memset1D on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hipMemsetParams memsetParams = {0};
  memsetParams.dst = dst;
  memsetParams.value = value;
  memsetParams.elementSize = valueSize;
  memsetParams.width = sizeBytes / valueSize;
  memsetParams.height = 1;

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipError_t status = ihipGraphAddMemsetNode(
      &pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
      s->GetLastCapturedNodes().size(), &memsetParams, true, 1, 1, 1, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemset2DAsync(hipStream_t& stream, void*& dst, size_t& pitch, int& value,
                                   size_t& width, size_t& height) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memset2D on stream : %p", stream);
  hipMemsetParams memsetParams = {0};
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  memsetParams.dst = dst;
  memsetParams.value = value;
  memsetParams.width = width;
  memsetParams.height = height;
  memsetParams.pitch = pitch;
  memsetParams.elementSize = 1;
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipError_t status = ihipGraphAddMemsetNode(
      &pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
      s->GetLastCapturedNodes().size(), &memsetParams, true, 1, 1, 1, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemset3DAsync(hipStream_t& stream, hipPitchedPtr& pitchedDevPtr, int& value,
                                   hipExtent& extent) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node Memset3D on stream : %p", stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  // In this case no work should be done
  if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
    return hipSuccess;
  }

  hipMemsetParams memsetParams = {0};
  memsetParams.dst = pitchedDevPtr.ptr;
  memsetParams.value = value;
  memsetParams.width = extent.width;
  memsetParams.height = extent.height;
  memsetParams.pitch = pitchedDevPtr.pitch;
  memsetParams.elementSize = 1;
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode;
  hipError_t status =
      ihipGraphAddMemsetNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &memsetParams, true, extent.depth,
                             pitchedDevPtr.xsize, pitchedDevPtr.ysize, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipLaunchHostFunc(hipStream_t& stream, hipHostFn_t& fn, void*& userData) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node HostFunction launch on stream : %p", stream);
  if (fn == nullptr) {
    return hipErrorInvalidValue;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hipHostNodeParams hostParams = {0};
  hostParams.fn = fn;
  hostParams.userData = userData;
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode* pGraphNode = new hip::GraphHostNode(&hostParams);
  hipError_t status =
      ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                       s->GetLastCapturedNodes().size(), true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

// ================================================================================================
hipError_t capturehipMallocAsync(hipStream_t stream, hipMemPool_t mem_pool, size_t size,
                                 void** dev_ptr) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node MallocAsync on stream : %p", stream);
  auto s = reinterpret_cast<hip::Stream*>(stream);
  auto mpool = reinterpret_cast<hip::MemoryPool*>(mem_pool);

  hipMemAllocNodeParams node_params{};

  node_params.poolProps.allocType = hipMemAllocationTypePinned;
  node_params.poolProps.location.id = mpool->Device()->deviceId();
  node_params.poolProps.location.type = hipMemLocationTypeDevice;

  std::vector<hipMemAccessDesc> descs;
  for (const auto device : g_devices) {
    hipMemLocation location{hipMemLocationTypeDevice, device->deviceId()};
    hipMemAccessFlags flags{};
    mpool->GetAccess(device, &flags);
    descs.push_back({location, flags});
  }

  node_params.accessDescs = &descs[0];
  node_params.accessDescCount = descs.size();
  node_params.bytesize = size;

  auto mem_alloc_node = new hip::GraphMemAllocNode(&node_params);
  auto status =
      ihipGraphAddNode(mem_alloc_node, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                       s->GetLastCapturedNodes().size(), true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  // Without VM runtime executes the node during capture, so it can return a valid device pointer
  *dev_ptr = (HIP_MEM_POOL_USE_VM) ? mem_alloc_node->ReserveAddress() : mem_alloc_node->Execute(s);
  s->SetLastCapturedNode(mem_alloc_node);

  return hipSuccess;
}

// ================================================================================================
hipError_t capturehipFreeAsync(hipStream_t stream, void* dev_ptr) {
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API,
          "[hipGraph] Current capture node FreeAsync on stream : %p", stream);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  auto mem_free_node = new hip::GraphMemFreeNode(dev_ptr);
  auto status =
      ihipGraphAddNode(mem_free_node, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                       s->GetLastCapturedNodes().size(), true, s->DeviceId());
  if (status != hipSuccess) {
    return status;
  }
  // Execute the node without VM support, so runtime can release memory into cache
  if (!HIP_MEM_POOL_USE_VM) {
    mem_free_node->Execute(s);
  }
  s->SetLastCapturedNode(mem_free_node);
  return hipSuccess;
}

// ================================================================================================
hipError_t hipStreamIsCapturing_common(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  if (pCaptureStatus == nullptr) {
    return hipErrorInvalidValue;
  }
  getStreamPerThread(stream);
  if (hip::Stream::StreamCaptureBlocking() == true &&
      (stream == nullptr || stream == hipStreamLegacy)) {
    return hipErrorStreamCaptureImplicit;
  }
  if (stream == nullptr || stream == hipStreamLegacy) {
    *pCaptureStatus = hipStreamCaptureStatusNone;
  } else {
    *pCaptureStatus = reinterpret_cast<hip::Stream*>(stream)->GetCaptureStatus();
  }
  return hipSuccess;
}

hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  HIP_INIT_API(hipStreamIsCapturing, stream, pCaptureStatus);
  HIP_RETURN(hipStreamIsCapturing_common(stream, pCaptureStatus));
}

hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus) {
  HIP_INIT_API(hipStreamIsCapturing, stream, pCaptureStatus);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamIsCapturing_common(stream, pCaptureStatus));
}

hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  HIP_INIT_API(hipThreadExchangeStreamCaptureMode, mode);

  if (mode == nullptr || *mode < hipStreamCaptureModeGlobal ||
      *mode > hipStreamCaptureModeRelaxed) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  auto oldMode = hip::tls.stream_capture_mode_;
  hip::tls.stream_capture_mode_ = *mode;
  *mode = oldMode;

  HIP_RETURN_DURATION(hipSuccess);
}

hipError_t hipStreamBeginCapture_common(hipStream_t stream, hipStreamCaptureMode mode,
                                        hipGraph_t graph = nullptr) {
  getStreamPerThread(stream);
  // capture cannot be initiated on legacy stream
  if (stream == nullptr || stream == hipStreamLegacy) {
    return hipErrorStreamCaptureUnsupported;
  }
  if (mode < hipStreamCaptureModeGlobal || mode > hipStreamCaptureModeRelaxed) {
    return hipErrorInvalidValue;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  // It can be initiated if the stream is not already in capture mode
  if (s->GetCaptureStatus() == hipStreamCaptureStatusActive) {
    return hipErrorIllegalState;
  }
  if (graph == nullptr) {
    s->SetCaptureGraph(new hip::Graph(s->GetDevice()));
  } else {
    s->SetCaptureGraph(reinterpret_cast<hip::Graph*>(graph));
  }
  s->SetCaptureId();
  s->SetCaptureMode(mode);
  s->SetOriginStream();
  if (mode != hipStreamCaptureModeRelaxed) {
    hip::tls.capture_streams_.push_back(s);
  }
  if (mode == hipStreamCaptureModeGlobal) {
    amd::ScopedLock lock(g_captureStreamsLock);
    g_captureStreams.push_back(s);
  }
  {
    amd::ScopedLock lock(g_streamSetLock);
    g_allCapturingStreams.insert(s);
  }
  return hipSuccess;
}

hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  HIP_INIT_API(hipStreamBeginCapture, stream, mode);
  HIP_RETURN_DURATION(hipStreamBeginCapture_common(stream, mode));
}

hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph,
                                        const hipGraphNode_t* dependencies,
                                        const hipGraphEdgeData* dependencyData,
                                        size_t numDependencies, hipStreamCaptureMode mode) {
  HIP_INIT_API(hipStreamBeginCapture, stream, graph, dependencies, dependencyData, numDependencies,
               mode);
  if (dependencyData != nullptr) {
    return hipErrorNotSupported;
  } else if (graph == nullptr) {
    return hipErrorInvalidValue;
  } else if (dependencies == nullptr && numDependencies != 0) {
    return hipErrorInvalidValue;
  } else if (dependencies != nullptr && numDependencies == 0) {
    return hipErrorInvalidValue;
  }
  hip::Graph* g = reinterpret_cast<hip::Graph*>(graph);
  const std::vector<Node> nodes = g->GetNodes();
  for (auto& node : nodes) {
    g->AddManualNodeDuringCapture(node);
  }
  hipError_t status = hipStreamBeginCapture_common(stream, mode, graph);
  if (status != hipSuccess) {
    HIP_RETURN_DURATION(status);
  }
  if (dependencies != nullptr) {
    status = ihipStreamUpdateCaptureDependencies(stream, const_cast<hipGraphNode_t*>(dependencies),
                                                 numDependencies, hipStreamSetCaptureDependencies);
  }
  HIP_RETURN_DURATION(status);
}

hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode) {
  HIP_INIT_API(hipStreamBeginCapture, stream, mode);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(hipStreamBeginCapture_common(stream, mode));
}

hipError_t hipStreamEndCapture_common(hipStream_t stream, hip::Graph** pGraph) {
  if (pGraph == nullptr) {
    return hipErrorInvalidValue;
  }
  if (stream == nullptr || stream == hipStreamLegacy) {
    return hipErrorIllegalState;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  // Capture status must be active before endCapture can be initiated
  if (s->GetCaptureStatus() == hipStreamCaptureStatusNone) {
    return hipErrorIllegalState;
  }
  // Capture must be ended on the same stream in which it was initiated
  if (!s->IsOriginStream()) {
    return hipErrorStreamCaptureUnmatched;
  }
  // If mode is not hipStreamCaptureModeRelaxed, hipStreamEndCapture must be called on the stream
  // from the same thread
  const auto& it = std::find(hip::tls.capture_streams_.begin(), hip::tls.capture_streams_.end(), s);
  if (s->GetCaptureMode() != hipStreamCaptureModeRelaxed) {
    if (it == hip::tls.capture_streams_.end()) {
      return hipErrorStreamCaptureWrongThread;
    }
    hip::tls.capture_streams_.erase(it);
  }
  if (s->GetCaptureMode() == hipStreamCaptureModeGlobal) {
    amd::ScopedLock lock(g_captureStreamsLock);
    g_captureStreams.erase(std::find(g_captureStreams.begin(), g_captureStreams.end(), s));
  }
  {
    amd::ScopedLock lock(g_streamSetLock);
    g_allCapturingStreams.erase(
        std::find(g_allCapturingStreams.begin(), g_allCapturingStreams.end(), s));
  }
  // If capture was invalidated, due to a violation of the rules of stream capture
  if (s->GetCaptureStatus() == hipStreamCaptureStatusInvalidated) {
    *pGraph = nullptr;
    // When capture is invalidated, graph should be deleted, otherwise it leaks
    s->ReleaseCaptureGraph();

    return hipErrorStreamCaptureInvalidated;
  }

  // Add temporary node to check if all parallel streams have joined
  hip::GraphNode* pGraphNode;
  pGraphNode = reinterpret_cast<hip::GraphNode*>(new hip::GraphEmptyNode());
  hipError_t status =
      ihipGraphAddNode(pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                       s->GetLastCapturedNodes().size());
  if (s->GetCaptureGraph()->GetLeafNodeCount() > 1) {
    std::vector<hip::GraphNode*> leafNodes = s->GetCaptureGraph()->GetLeafNodes();
    // Manually added nodes doesnt result in hipErrorStreamCaptureUnjoined. Remove them from leaf
    // nodes.
    std::unordered_set<hip::GraphNode*> nodes = s->GetCaptureGraph()->GetManualNodesDuringCapture();
    for (auto node : nodes) {
      const auto& fnode = std::find(leafNodes.begin(), leafNodes.end(), node);
      if (fnode != leafNodes.end()) {
        leafNodes.erase(fnode);
      }
    }
    // Nodes that are removed from the dependency set via API hipStreamUpdateCaptureDependencies do
    // not result in hipErrorStreamCaptureUnjoined
    const std::vector<hip::GraphNode*>& removedDepNodes = s->GetRemovedDependencies();
    bool foundInRemovedDep = false;
    for (auto leafNode : leafNodes) {
      for (auto node : removedDepNodes) {
        if (node == leafNode) {
          foundInRemovedDep = true;
        }
      }
    }
    // remove temporary node
    s->GetCaptureGraph()->RemoveNode(pGraphNode);
    s->GetCaptureGraph()->RemoveManualNodesDuringCapture();
    if (leafNodes.size() > 1 && foundInRemovedDep == false) {
      // Release created graph as it can't be retrieved anymore
      s->ReleaseCaptureGraph();
      return hipErrorStreamCaptureUnjoined;
    }
  } else {
    // remove temporary node
    s->GetCaptureGraph()->RemoveNode(pGraphNode);
  }

  *pGraph = s->GetCaptureGraph();
  // end capture on all streams/events part of graph capture
  return s->EndCapture();
}

hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  HIP_INIT_API(hipStreamEndCapture, stream, pGraph);
  hip::Graph* graph;
  hipError_t status = hipStreamEndCapture_common(stream, &graph);
  if (pGraph != nullptr) {
    *pGraph = reinterpret_cast<hipGraph_t>(graph);
  }
  HIP_RETURN(status);
}

hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t* pGraph) {
  HIP_INIT_API(hipStreamEndCapture, stream, pGraph);
  PER_THREAD_DEFAULT_STREAM(stream);
  hip::Graph* graph;
  hipError_t status = hipStreamEndCapture_common(stream, &graph);
  if (pGraph != nullptr) {
    *pGraph = reinterpret_cast<hipGraph_t>(graph);
  }
  HIP_RETURN(status);
}

hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  HIP_INIT_API(hipGraphCreate, pGraph, flags);
  if ((pGraph == nullptr) || (flags != 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pGraph = reinterpret_cast<hipGraph_t>(new hip::Graph(hip::getCurrentDevice()));
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphDestroy(hipGraph_t graph) {
  HIP_INIT_API(hipGraphDestroy, graph);
  if (graph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Graph* g = reinterpret_cast<hip::Graph*>(graph);
  // if graph is not valid its destroyed already
  if (!hip::Graph::isGraphValid(g)) {
    HIP_RETURN(hipErrorIllegalState);
  }
  delete g;
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphAddKernelNode, pGraphNode, graph, pDependencies, numDependencies,
               pNodeParams);
  if (pGraphNode == nullptr || graph == nullptr || pNodeParams == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pNodeParams->func == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node;
  hipError_t status =
      ihipGraphAddKernelNode(&node, reinterpret_cast<hip::Graph*>(graph),
                             reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                             numDependencies, pNodeParams, nullptr, false);
  *pGraphNode = reinterpret_cast<hipGraphNode*>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams) {
  HIP_INIT_API(hipGraphAddMemcpyNode, pGraphNode, graph, pDependencies, numDependencies,
               pCopyParams);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node;
  hipError_t status = ihipGraphAddMemcpyNode(
      &node, reinterpret_cast<hip::Graph*>(graph),
      reinterpret_cast<hip::GraphNode* const*>(pDependencies), numDependencies, pCopyParams, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const HIP_MEMCPY3D* copyParams, hipCtx_t ctx) {
  HIP_INIT_API(hipDrvGraphAddMemcpyNode, phGraphNode, hGraph, dependencies, numDependencies,
               copyParams, ctx);
  if (phGraphNode == nullptr || hGraph == nullptr ||
      (numDependencies > 0 && dependencies == nullptr) || ctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node;
  hipError_t status =
      ihipDrvGraphAddMemcpyNode(&node, reinterpret_cast<hip::Graph*>(hGraph),
                                reinterpret_cast<hip::GraphNode* const*>(dependencies),
                                numDependencies, copyParams, ctx, false);
  *phGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphAddMemcpyNode1D, pGraphNode, graph, pDependencies, numDependencies, dst, src,
               count, kind);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node;
  hipError_t status =
      ihipGraphAddMemcpyNode1D(&node, reinterpret_cast<hip::Graph*>(graph),
                               reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                               numDependencies, dst, src, count, kind, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParams1D, node, dst, src, count, kind);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (!hip::GraphNode::isNodeValid(n) || dst == nullptr || src == nullptr || count == 0 ||
      src == dst) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(reinterpret_cast<hip::GraphMemcpyNode1D*>(n)->SetParams(dst, src, count, kind));
}

hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParams1D, hGraphExec, node, dst, src, count, kind);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (hGraphExec == nullptr || !hip::GraphNode::isNodeValid(n) || dst == nullptr ||
      src == nullptr || count == 0 || src == dst || n->GetType() != hipGraphNodeTypeMemcpy) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphNode*>(
      reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n));
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipMemcpyKind oldkind = reinterpret_cast<hip::GraphMemcpyNode1D*>(clonedNode)->GetMemcpyKind();
  if (oldkind != kind) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipError_t status =
      reinterpret_cast<hip::GraphMemcpyNode1D*>(clonedNode)->SetParams(dst, src, count, kind);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    status = reinterpret_cast<hip::GraphExec*>(hGraphExec)
                 ->UpdateAQLPacket(reinterpret_cast<hip::GraphKernelNode*>(clonedNode));
  }
  HIP_RETURN(status);
}

hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams) {
  HIP_INIT_API(hipGraphAddMemsetNode, pGraphNode, graph, pDependencies, numDependencies,
               pMemsetParams);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node;
  hipError_t status =
      ihipGraphAddMemsetNode(&node, reinterpret_cast<hip::Graph*>(graph),
                             reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                             numDependencies, pMemsetParams, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  HIP_INIT_API(hipDrvGraphAddMemsetNode, phGraphNode, hGraph, dependencies, numDependencies,
               memsetParams, ctx);
  if (phGraphNode == nullptr || hGraph == nullptr ||
      (numDependencies > 0 && dependencies == nullptr) || memsetParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node;
  hipMemsetParams pmemsetParams;
  pmemsetParams.dst = reinterpret_cast<void*>(memsetParams->dst);
  pmemsetParams.elementSize = memsetParams->elementSize;
  pmemsetParams.height = memsetParams->height;
  pmemsetParams.pitch = memsetParams->pitch;
  pmemsetParams.value = memsetParams->value;
  pmemsetParams.width = memsetParams->width;
  hipError_t status = ihipGraphAddMemsetNode(&node, reinterpret_cast<hip::Graph*>(hGraph),
                                             reinterpret_cast<hip::GraphNode* const*>(dependencies),
                                             numDependencies, &pmemsetParams, false);
  *phGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies) {
  HIP_INIT_API(hipGraphAddEmptyNode, pGraphNode, graph, pDependencies, numDependencies);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node = new hip::GraphEmptyNode();
  hipError_t status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                       reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                       numDependencies, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph) {
  HIP_INIT_API(hipGraphAddChildGraphNode, pGraphNode, pDependencies, numDependencies, childGraph);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || childGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node = new hip::ChildGraphNode(reinterpret_cast<hip::Graph*>(childGraph));
  hipError_t status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                       reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                       numDependencies, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t ihipGraphInstantiate(hip::GraphExec** pGraphExec, hip::Graph* graph,
                                uint64_t flags = 0) {
  if (pGraphExec == nullptr || graph == nullptr) {
    return hipErrorInvalidValue;
  }
  if (graph->IsGraphInstantiated() == true) {
    for (auto node : graph->GetNodes()) {
      if ((node->GetType() == hipGraphNodeTypeMemAlloc) ||
          (node->GetType() == hipGraphNodeTypeMemFree)) {
        return hipErrorNotSupported;
      }
    }
  }
  *pGraphExec = new hip::GraphExec(flags);
  if (*pGraphExec == nullptr) {
    return hipErrorOutOfMemory;
  }
  graph->clone(*pGraphExec, true);
  (*pGraphExec)->ScheduleNodes();
  if (false == (*pGraphExec)->TopologicalOrder()) {
    delete *pGraphExec;
    return hipErrorInvalidValue;
  }
  graph->SetGraphInstantiated(true);
  if (DEBUG_HIP_GRAPH_DOT_PRINT) {
    static int i = 1;
    std::string filename =
        "graph_" + std::to_string(amd::Os::getProcessId()) + "_dot_print_" + std::to_string(i++);
    hipError_t status = ihipGraphDebugDotPrint(*pGraphExec, filename.c_str(), 0);
    if (status == hipSuccess) {
      LogPrintfInfo("[hipGraph] graph dump:%s", filename.c_str());
    }
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    (*pGraphExec)->SetKernelArgManager(new hip::GraphKernelArgManager());
  }
  return (*pGraphExec)->Init();
}

hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  HIP_INIT_API(hipGraphInstantiate, pGraphExec, graph);
  if (pGraphExec == nullptr || graph == nullptr) {
    return hipErrorInvalidValue;
  }
  hip::GraphExec* ge;
  hipError_t status = ihipGraphInstantiate(&ge, reinterpret_cast<hip::Graph*>(graph));
  if (status == hipSuccess) {
    *pGraphExec = reinterpret_cast<hipGraphExec_t>(ge);
  }
  HIP_RETURN(status);
}

hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags = 0) {
  HIP_INIT_API(hipGraphInstantiateWithFlags, pGraphExec, graph, flags);
  if (pGraphExec == nullptr || graph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // invalid flag check
  if (flags != 0 && flags != hipGraphInstantiateFlagAutoFreeOnLaunch &&
      flags != hipGraphInstantiateFlagUseNodePriority) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::GraphExec* ge;
  hipError_t status = ihipGraphInstantiate(&ge, reinterpret_cast<hip::Graph*>(graph), flags);
  *pGraphExec = reinterpret_cast<hipGraphExec_t>(ge);
  HIP_RETURN(status);
}

hipError_t hipGraphInstantiateWithParams(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                         hipGraphInstantiateParams* instantiateParams) {
  HIP_INIT_API(hipGraphInstantiateWithParams, pGraphExec, graph, instantiateParams);
  if (pGraphExec == nullptr || graph == nullptr || instantiateParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  unsigned long long flags = instantiateParams->flags;

  if (flags != 0 && flags != hipGraphInstantiateFlagAutoFreeOnLaunch &&
      flags != hipGraphInstantiateFlagUpload && flags != hipGraphInstantiateFlagDeviceLaunch &&
      flags != hipGraphInstantiateFlagUseNodePriority) {
    instantiateParams->result_out = hipGraphInstantiateError;
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::GraphExec* ge;
  hipError_t status = ihipGraphInstantiate(&ge, reinterpret_cast<hip::Graph*>(graph), flags);
  *pGraphExec = reinterpret_cast<hipGraphExec_t>(ge);
  if (status != hipSuccess) {
    instantiateParams->result_out = hipGraphInstantiateError;
    HIP_RETURN(status);
  }

  instantiateParams->result_out = hipGraphInstantiateSuccess;
  instantiateParams->errNode_out = nullptr;

  if (flags == hipGraphInstantiateFlagUpload) {
    hipError_t status = ihipGraphUpload(*pGraphExec, instantiateParams->uploadStream);
    HIP_RETURN(status);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecDestroy(hipGraphExec_t pGraphExec) {
  HIP_INIT_API(hipGraphExecDestroy, pGraphExec);
  if (pGraphExec == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphExec* ge = reinterpret_cast<hip::GraphExec*>(pGraphExec);
  ge->release();
  amd::ScopedLock lock(GraphExec::graphExecSetLock_);
  GraphExec::graphExecSet_.erase(ge);
  HIP_RETURN(hipSuccess);
}

hipError_t ihipGraphLaunch(hip::GraphExec* graphExec, hipStream_t stream) {
  getStreamPerThread(stream);
  hip::Stream* launch_stream = hip::getStream(stream);
  return graphExec->Run(launch_stream);
}

hipError_t hipGraphLaunch_common(hip::GraphExec* graphExec, hipStream_t stream) {
  if (graphExec == nullptr || !hip::GraphExec::isGraphExecValid(graphExec)) {
    return hipErrorInvalidValue;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  if (graphExec->GetNodeCount() == 0) {
    return hipSuccess;
  }
  return ihipGraphLaunch(graphExec, stream);
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  HIP_INIT_API(hipGraphLaunch, graphExec, stream);
  HIP_RETURN_DURATION(hipGraphLaunch_common(reinterpret_cast<hip::GraphExec*>(graphExec), stream));
}

hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream) {
  HIP_INIT_API(hipGraphLaunch, graphExec, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN_DURATION(hipGraphLaunch_common(reinterpret_cast<hip::GraphExec*>(graphExec), stream));
}

hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
  HIP_INIT_API(hipGraphGetNodes, graph, nodes, numNodes);
  if (graph == nullptr || numNodes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  std::vector<hip::GraphNode*> graphNodes;
  if (false == reinterpret_cast<hip::Graph*>(graph)->TopologicalOrder(graphNodes)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (nodes == nullptr) {
    *numNodes = graphNodes.size();
    HIP_RETURN(hipSuccess);
  } else if (*numNodes <= graphNodes.size()) {
    for (int i = 0; i < *numNodes; i++) {
      nodes[i] = reinterpret_cast<hipGraphNode_t>(graphNodes[i]);
    }
  } else {
    for (int i = 0; i < graphNodes.size(); i++) {
      nodes[i] = reinterpret_cast<hipGraphNode_t>(graphNodes[i]);
    }
    for (int i = graphNodes.size(); i < *numNodes; i++) {
      nodes[i] = nullptr;
    }
    *numNodes = graphNodes.size();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes) {
  HIP_INIT_API(hipGraphGetRootNodes, graph, pRootNodes, pNumRootNodes);

  if (graph == nullptr || pNumRootNodes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<hip::GraphNode*> nodes = reinterpret_cast<hip::Graph*>(graph)->GetRootNodes();
  if (pRootNodes == nullptr) {
    *pNumRootNodes = nodes.size();
    HIP_RETURN(hipSuccess);
  } else if (*pNumRootNodes <= nodes.size()) {
    for (int i = 0; i < *pNumRootNodes; i++) {
      pRootNodes[i] = reinterpret_cast<hipGraphNode_t>(nodes[i]);
    }
  } else {
    for (int i = 0; i < nodes.size(); i++) {
      pRootNodes[i] = reinterpret_cast<hipGraphNode_t>(nodes[i]);
    }
    for (int i = nodes.size(); i < *pNumRootNodes; i++) {
      pRootNodes[i] = nullptr;
    }
    *pNumRootNodes = nodes.size();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphKernelNodeGetParams, node, pNodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) ||
      pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (reinterpret_cast<hip::GraphNode*>(node)->GetType() != hipGraphNodeTypeKernel) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphKernelNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node,
                                       const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphKernelNodeSetParams, node, pNodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) ||
      pNodeParams == nullptr || pNodeParams->func == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (reinterpret_cast<hip::GraphNode*>(node)->GetType() != hipGraphNodeTypeKernel) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphKernelNode*>(node)->SetParams(pNodeParams));
}

hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphMemcpyNodeGetParams, node, pNodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) ||
      pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphMemcpyNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          const hipKernelNodeAttrValue* value) {
  HIP_INIT_API(hipGraphKernelNodeSetAttribute, hNode, attr, value);
  if (hNode == nullptr || value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (attr != hipKernelNodeAttributeAccessPolicyWindow &&
      attr != hipKernelNodeAttributeCooperative && attr != hipLaunchAttributePriority) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (reinterpret_cast<hip::GraphNode*>(hNode)->GetType() != hipGraphNodeTypeKernel) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(reinterpret_cast<hip::GraphKernelNode*>(hNode)->SetAttrParams(attr, value));
}

hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          hipKernelNodeAttrValue* value) {
  HIP_INIT_API(hipGraphKernelNodeGetAttribute, hNode, attr, value);
  if (hNode == nullptr || value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (attr != hipKernelNodeAttributeAccessPolicyWindow &&
      attr != hipKernelNodeAttributeCooperative && attr != hipLaunchAttributePriority) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (reinterpret_cast<hip::GraphNode*>(hNode)->GetType() != hipGraphNodeTypeKernel) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(reinterpret_cast<hip::GraphKernelNode*>(hNode)->GetAttrParams(attr, value));
}

hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParams, node, pNodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) ||
      pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphMemcpyNode*>(node)->SetParams(pNodeParams));
}

hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParams, hGraphExec, node, pNodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (hGraphExec == nullptr || !hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(n)) ||
      n->GetType() != hipGraphNodeTypeMemcpy) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (ihipMemcpy3D_validate(pNodeParams) != hipSuccess) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Check if pNodeParams passed is a empty struct
  if (((pNodeParams->srcArray == 0) && (pNodeParams->srcPtr.ptr == nullptr)) ||
      ((pNodeParams->dstArray == 0) && (pNodeParams->dstPtr.ptr == nullptr))) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipMemcpyKind oldkind = reinterpret_cast<hip::GraphMemcpyNode*>(clonedNode)->GetMemcpyKind();
  hipMemcpyKind newkind = pNodeParams->kind;
  if (oldkind != newkind) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipError_t status = reinterpret_cast<hip::GraphMemcpyNode*>(clonedNode)->SetParams(pNodeParams);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    status = reinterpret_cast<hip::GraphExec*>(hGraphExec)
                 ->UpdateAQLPacket(reinterpret_cast<hip::GraphKernelNode*>(clonedNode));
  }
  HIP_RETURN(status);
}

hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams) {
  HIP_INIT_API(hipGraphMemsetNodeGetParams, node, pNodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) ||
      pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphMemsetNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams) {
  HIP_INIT_API(hipGraphMemsetNodeSetParams, node, pNodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) ||
      pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (pNodeParams->height > 1 &&
      pNodeParams->pitch < (pNodeParams->width * pNodeParams->elementSize)) {
    return hipErrorInvalidValue;
  }
  HIP_RETURN(reinterpret_cast<hip::GraphMemsetNode*>(node)->SetParams(pNodeParams));
}

hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams) {
  HIP_INIT_API(hipGraphExecMemsetNodeSetParams, hGraphExec, node, pNodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);

  if (hGraphExec == nullptr || !hip::GraphNode::isNodeValid(n) || pNodeParams == nullptr ||
      pNodeParams->dst == nullptr || n->GetType() != hipGraphNodeTypeMemset) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (ihipGraphMemsetParams_validate(pNodeParams) != hipSuccess) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipError_t status =
      reinterpret_cast<hip::GraphMemsetNode*>(clonedNode)->SetParams(pNodeParams, true);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    status = reinterpret_cast<hip::GraphExec*>(hGraphExec)
                 ->UpdateAQLPacket(reinterpret_cast<hip::GraphKernelNode*>(clonedNode));
  }
  HIP_RETURN(status);
}

hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies) {
  HIP_INIT_API(hipGraphAddDependencies, graph, from, to, numDependencies);
  if (graph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (numDependencies == 0) {
    HIP_RETURN(hipSuccess);
  } else if (from == nullptr || to == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* const* fromNode = reinterpret_cast<hip::GraphNode* const*>(from);
  hip::GraphNode* const* toNode = reinterpret_cast<hip::GraphNode* const*>(to);
  hip::Graph* g = reinterpret_cast<hip::Graph*>(graph);
  for (size_t i = 0; i < numDependencies; i++) {
    // When the same node is specified for both fromNode and toNode
    if (fromNode[i] == nullptr || toNode[i] == nullptr || fromNode[i] == toNode[i] ||
        !hip::GraphNode::isNodeValid(toNode[i]) || !hip::GraphNode::isNodeValid(fromNode[i]) ||
        // making sure the nodes belong toNode the graph
        toNode[i]->GetParentGraph() != g || fromNode[i]->GetParentGraph() != g) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  for (size_t i = 0; i < numDependencies; i++) {
    // When the same edge added fromNode->toNode return invalid value
    const std::vector<hip::GraphNode*>& edges = fromNode[i]->GetEdges();
    for (auto edge : edges) {
      if (edge == toNode[i]) {
        HIP_RETURN(hipErrorInvalidValue);
      }
    }
    fromNode[i]->AddEdgeDep(toNode[i]);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphExecKernelNodeSetParams, hGraphExec, node, pNodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (hGraphExec == nullptr || !hip::GraphNode::isNodeValid(n) || pNodeParams == nullptr ||
      pNodeParams->func == nullptr || n->GetType() != hipGraphNodeTypeKernel) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipError_t status = reinterpret_cast<hip::GraphKernelNode*>(clonedNode)->SetParams(pNodeParams);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    status = reinterpret_cast<hip::GraphExec*>(hGraphExec)
                 ->UpdateAQLPacket(reinterpret_cast<hip::GraphKernelNode*>(clonedNode));
  }
  HIP_RETURN(status);
}

hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph) {
  HIP_INIT_API(hipGraphChildGraphNodeGetGraph, node, pGraph);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) || pGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Graph* g = reinterpret_cast<hip::GraphNode*>(node)->GetChildGraph();
  *pGraph = reinterpret_cast<hipGraph_t>(g);
  if (*pGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t validateChildGraphNodeSetParams(hip::GraphNode* n, hip::Graph* cg, bool exec = true) {
  if (cg == nullptr || n == nullptr || !hip::GraphNode::isNodeValid(n) ||
      !hip::Graph::isGraphValid(cg) || n->GetType() != hipGraphNodeTypeGraph) {
    return hipErrorInvalidValue;
  }
  // compare with parent graph fron cloned and original node
  if (cg == n->GetParentGraph()->getOriginalGraph() || cg == n->GetParentGraph()) {
    return hipErrorUnknown;
  }

  if (exec) {  // validation only required for ExecnNodeSetParams
    // Validate whether the topology of node and childGraph matches
    std::vector<hip::GraphNode*> childGraphNodes1;
    n->TopologicalOrder(childGraphNodes1);

    std::vector<hip::GraphNode*> childGraphNodes2;
    cg->TopologicalOrder(childGraphNodes2);

    if (childGraphNodes1.size() != childGraphNodes2.size()) {
      return hipErrorUnknown;
    }
    // Validate if the node insertion order matches
    else {
      for (std::vector<hip::GraphNode*>::size_type i = 0; i != childGraphNodes1.size(); i++) {
        if (childGraphNodes1[i]->GetType() != childGraphNodes2[i]->GetType()) {
          return hipErrorUnknown;
        }
      }
    }
  }
  return hipSuccess;
}

hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph) {
  HIP_INIT_API(hipGraphExecChildGraphNodeSetParams, hGraphExec, node, childGraph);

  if (hGraphExec == nullptr || childGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  hip::Graph* cg = reinterpret_cast<hip::Graph*>(childGraph);

  hipError_t status = validateChildGraphNodeSetParams(n, cg);
  if (status != hipSuccess) {
    return status;
  }

  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  status = reinterpret_cast<hip::ChildGraphNode*>(clonedNode)->SetParams(cg);
  if (status != hipSuccess) {
    return status;
  }
  if (reinterpret_cast<hip::ChildGraphNode*>(clonedNode)->GetGraphCaptureStatus()) {
    std::vector<hip::GraphNode*> childGraphNodes;
    reinterpret_cast<hip::ChildGraphNode*>(clonedNode)->TopologicalOrder(childGraphNodes);
    for (std::vector<hip::GraphNode*>::size_type i = 0; i != childGraphNodes.size(); i++) {
      if (childGraphNodes[i]->GraphCaptureEnabled()) {
        status = reinterpret_cast<hip::ChildGraphNode*>(clonedNode)
                     ->UpdateAQLPacket(reinterpret_cast<hip::GraphKernelNode*>(childGraphNodes[i]));
        if (status != hipSuccess) {
          return status;
        }
      }
    }
  }
  return status;
}

hipError_t hipStreamGetCaptureInfo_common(hipStream_t stream,
                                          hipStreamCaptureStatus* pCaptureStatus,
                                          unsigned long long* pId) {
  if (pCaptureStatus == nullptr) {
    return hipErrorInvalidValue;
  }
  getStreamPerThread(stream);
  if (hip::Stream::StreamCaptureBlocking() == true &&
      (stream == nullptr || stream == hipStreamLegacy)) {
    return hipErrorStreamCaptureImplicit;
  }
  if (stream == nullptr || stream == hipStreamLegacy) {
    *pCaptureStatus = hipStreamCaptureStatusNone;
    return hipSuccess;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  *pCaptureStatus = s->GetCaptureStatus();
  if (*pCaptureStatus == hipStreamCaptureStatusActive && pId != nullptr) {
    *pId = s->GetCaptureID();
  }
  return hipSuccess;
}

hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId) {
  HIP_INIT_API(hipStreamGetCaptureInfo, stream, pCaptureStatus, pId);
  HIP_RETURN(hipStreamGetCaptureInfo_common(stream, pCaptureStatus, pId));
}

hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                       unsigned long long* pId) {
  HIP_INIT_API(hipStreamGetCaptureInfo, stream, pCaptureStatus, pId);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamGetCaptureInfo_common(stream, pCaptureStatus, pId));
}

hipError_t hipStreamGetCaptureInfo_v2_common(hipStream_t stream,
                                             hipStreamCaptureStatus* captureStatus_out,
                                             unsigned long long* id_out, hipGraph_t* graph_out,
                                             const hipGraphNode_t** dependencies_out,
                                             size_t* numDependencies_out) {
  if (captureStatus_out == nullptr) {
    return hipErrorInvalidValue;
  }
  if (hip::Stream::StreamCaptureBlocking() == true &&
      (stream == nullptr || stream == hipStreamLegacy)) {
    return hipErrorStreamCaptureImplicit;
  }
  if (stream == nullptr || stream == hipStreamLegacy) {
    *captureStatus_out = hipStreamCaptureStatusNone;
    return hipSuccess;
  }
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  *captureStatus_out = s->GetCaptureStatus();
  if (*captureStatus_out == hipStreamCaptureStatusActive) {
    if (id_out != nullptr) {
      *id_out = s->GetCaptureID();
    }
    if (graph_out != nullptr) {
      *graph_out = reinterpret_cast<hipGraph_t>(s->GetCaptureGraph());
    }
    if (dependencies_out != nullptr) {
      *dependencies_out = reinterpret_cast<hipGraphNode_t const*>(s->GetLastCapturedNodes().data());
    }
    if (numDependencies_out != nullptr) {
      *numDependencies_out = s->GetLastCapturedNodes().size();
    }
  }
  return hipSuccess;
}

hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out) {
  HIP_INIT_API(hipStreamGetCaptureInfo_v2, stream, captureStatus_out, id_out, graph_out,
               dependencies_out, numDependencies_out);
  HIP_RETURN(hipStreamGetCaptureInfo_v2_common(stream, captureStatus_out, id_out, graph_out,
                                               dependencies_out, numDependencies_out));
}

hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,
                                          hipStreamCaptureStatus* captureStatus_out,
                                          unsigned long long* id_out, hipGraph_t* graph_out,
                                          const hipGraphNode_t** dependencies_out,
                                          size_t* numDependencies_out) {
  HIP_INIT_API(hipStreamGetCaptureInfo_v2, stream, captureStatus_out, id_out, graph_out,
               dependencies_out, numDependencies_out);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipStreamGetCaptureInfo_v2_common(stream, captureStatus_out, id_out, graph_out,
                                               dependencies_out, numDependencies_out));
}

hipError_t ihipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                               size_t numDependencies, unsigned int flags) {
  HIP_INIT_API(hipStreamUpdateCaptureDependencies, stream, dependencies, numDependencies, flags);
  if (!hip::isValid(stream)) {
    HIP_RETURN(hipErrorContextIsDestroyed);
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::GraphNode** deps = reinterpret_cast<hip::GraphNode**>(dependencies);
  if (s->GetCaptureStatus() == hipStreamCaptureStatusNone) {
    HIP_RETURN(hipErrorIllegalState);
  }
  if ((s->GetCaptureGraph()->GetNodeCount() < numDependencies) ||
      (numDependencies > 0 && deps == nullptr) ||
      (flags != 0 && flags != hipStreamAddCaptureDependencies &&
       flags != hipStreamSetCaptureDependencies)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  std::vector<hip::GraphNode*> depNodes;
  const std::vector<hip::GraphNode*>& graphNodes = s->GetCaptureGraph()->GetNodes();
  for (int i = 0; i < numDependencies; i++) {
    if ((deps[i] == nullptr) ||
        std::find(std::begin(graphNodes), std::end(graphNodes), deps[i]) == std::end(graphNodes)) {
      HIP_RETURN(hipErrorInvalidValue);
    }
    depNodes.push_back(deps[i]);
  }
  if (flags == hipStreamAddCaptureDependencies) {
    s->AddCrossCapturedNode(depNodes);
  } else if (flags == hipStreamSetCaptureDependencies) {
    bool replace = true;
    s->AddCrossCapturedNode(depNodes, replace);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies, unsigned int flags) {
  HIP_INIT_API(hipStreamUpdateCaptureDependencies, stream, dependencies, numDependencies, flags);
  HIP_RETURN(ihipStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags));
}

hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies) {
  HIP_INIT_API(hipGraphRemoveDependencies, graph, from, to, numDependencies);
  if (graph == nullptr || (numDependencies > 0 && (from == nullptr || to == nullptr))) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* const* fromNode = reinterpret_cast<hip::GraphNode* const*>(from);
  hip::GraphNode* const* toNode = reinterpret_cast<hip::GraphNode* const*>(to);
  hip::Graph* g = reinterpret_cast<hip::Graph*>(graph);
  for (size_t i = 0; i < numDependencies; i++) {
    if (toNode[i]->GetParentGraph() != g || fromNode[i]->GetParentGraph() != g ||
        fromNode[i]->RemoveEdgeDep(toNode[i]) == false) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges) {
  HIP_INIT_API(hipGraphGetEdges, graph, from, to, numEdges);
  if (graph == nullptr || numEdges == nullptr || (from == nullptr && to != nullptr) ||
      (to == nullptr && from != nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<std::pair<hip::GraphNode*, hip::GraphNode*>> edges =
      reinterpret_cast<hip::Graph*>(graph)->GetEdges();
  // returns only the number of edges in numEdges when from and to are null
  if (from == nullptr && to == nullptr) {
    *numEdges = edges.size();
    HIP_RETURN(hipSuccess);
  } else if (*numEdges <= edges.size()) {
    for (int i = 0; i < *numEdges; i++) {
      from[i] = reinterpret_cast<hipGraphNode_t>(edges[i].first);
      to[i] = reinterpret_cast<hipGraphNode_t>(edges[i].second);
    }
  } else {
    for (int i = 0; i < edges.size(); i++) {
      from[i] = reinterpret_cast<hipGraphNode_t>(edges[i].first);
      to[i] = reinterpret_cast<hipGraphNode_t>(edges[i].second);
    }
    // If numEdges > actual number of edges, the remaining entries in from and to will be set to
    // NULL
    for (int i = edges.size(); i < *numEdges; i++) {
      from[i] = nullptr;
      to[i] = nullptr;
    }
    *numEdges = edges.size();
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies) {
  HIP_INIT_API(hipGraphNodeGetDependencies, node, pDependencies, pNumDependencies);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (!hip::GraphNode::isNodeValid(n) || pNumDependencies == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<hip::GraphNode*>& dependencies = n->GetDependencies();
  if (pDependencies == NULL) {
    *pNumDependencies = dependencies.size();
    HIP_RETURN(hipSuccess);
  } else if (*pNumDependencies <= dependencies.size()) {
    for (int i = 0; i < *pNumDependencies; i++) {
      pDependencies[i] = reinterpret_cast<hipGraphNode_t>(dependencies[i]);
    }
  } else {
    for (int i = 0; i < dependencies.size(); i++) {
      pDependencies[i] = reinterpret_cast<hipGraphNode_t>(dependencies[i]);
    }
    // pNumDependencies > actual number of dependencies, the remaining entries in pDependencies
    // will be set to NULL
    for (int i = dependencies.size(); i < *pNumDependencies; i++) {
      pDependencies[i] = nullptr;
    }
    *pNumDependencies = dependencies.size();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes) {
  HIP_INIT_API(hipGraphNodeGetDependentNodes, node, pDependentNodes, pNumDependentNodes);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (!hip::GraphNode::isNodeValid(n) || pNumDependentNodes == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const std::vector<hip::GraphNode*>& dependents = n->GetEdges();
  if (pDependentNodes == NULL) {
    *pNumDependentNodes = dependents.size();
    HIP_RETURN(hipSuccess);
  } else if (*pNumDependentNodes <= dependents.size()) {
    for (int i = 0; i < *pNumDependentNodes; i++) {
      pDependentNodes[i] = reinterpret_cast<hipGraphNode_t>(dependents[i]);
    }
  } else {
    for (int i = 0; i < dependents.size(); i++) {
      pDependentNodes[i] = reinterpret_cast<hipGraphNode_t>(dependents[i]);
    }
    // pNumDependentNodes > actual number of dependents, the remaining entries in pDependentNodes
    // will be set to NULL
    for (int i = dependents.size(); i < *pNumDependentNodes; i++) {
      pDependentNodes[i] = nullptr;
    }
    *pNumDependentNodes = dependents.size();
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType) {
  HIP_INIT_API(hipGraphNodeGetType, node, pType);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);

  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(n)) || pType == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *pType = n->GetType();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
  HIP_INIT_API(hipGraphDestroyNode, node);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);

  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(n))) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (n->GetType() == hipGraphNodeTypeMemAlloc || n->GetType() == hipGraphNodeTypeMemFree) {
    HIP_RETURN(hipErrorNotSupported);
  }
  // Remove the node from graph should takecare of updating edges of parent and deps of child nodes
  // as part of graph node destructor
  n->GetParentGraph()->RemoveNode(n);
  HIP_RETURN(hipSuccess);
}


hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
  HIP_INIT_API(hipGraphClone, pGraphClone, originalGraph);
  if (originalGraph == nullptr || pGraphClone == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Graph* g = reinterpret_cast<hip::Graph*>(originalGraph);
  if (!hip::Graph::isGraphValid(g)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  for (auto n : g->GetNodes()) {
    if (n->GetType() == hipGraphNodeTypeMemAlloc || n->GetType() == hipGraphNodeTypeMemFree) {
      HIP_RETURN(hipErrorNotSupported);
    }
  }
  *pGraphClone = reinterpret_cast<hipGraph_t>(g->clone());
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph) {
  HIP_INIT_API(hipGraphNodeFindInClone, pNode, originalNode, clonedGraph);
  if (pNode == nullptr || originalNode == nullptr || clonedGraph == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::Graph* iClonedGraph = reinterpret_cast<hip::Graph*>(clonedGraph);
  hip::GraphNode* iGraphnode = reinterpret_cast<hip::GraphNode*>(originalNode);
  if (iClonedGraph->getOriginalGraph() == nullptr ||
      // parent graph of orig node and original graph of cloned node should be same.
      iClonedGraph->getOriginalGraph() != iGraphnode->GetParentGraph()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  for (auto node : iClonedGraph->GetNodes()) {
    if (node->GetID() == iGraphnode->GetID()) {
      *pNode = reinterpret_cast<hipGraphNode_t>(node);
      HIP_RETURN(hipSuccess);
    }
  }
  HIP_RETURN(hipErrorInvalidValue);
}

hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphAddMemcpyNodeFromSymbol, pGraphNode, graph, pDependencies, numDependencies,
               dst, symbol, count, offset, kind);
  hip::Graph* g = reinterpret_cast<hip::Graph*>(graph);
  if (graph == nullptr || pGraphNode == nullptr || count == 0 ||
      (numDependencies > 0 && pDependencies == nullptr) || dst == nullptr ||
      !hip::Graph::isGraphValid(g)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;

  hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  hip::GraphNode* node = new hip::GraphMemcpyNodeFromSymbol(dst, symbol, count, offset, kind);
  status = ihipGraphAddNode(node, g, reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                            numDependencies, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParamsFromSymbol, node, dst, symbol, count, offset, kind);
  if (symbol == nullptr) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) || dst == nullptr ||
      count == 0 || symbol == dst) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(reinterpret_cast<hip::GraphMemcpyNodeFromSymbol*>(node)->SetParams(dst, symbol, count,
                                                                                offset, kind));
}

hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParamsFromSymbol, hGraphExec, node, dst, symbol, count,
               offset, kind);
  if (symbol == nullptr) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (hGraphExec == nullptr || !hip::GraphNode::isNodeValid(n) || dst == nullptr || count == 0 ||
      symbol == dst) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipMemcpyKind oldkind =
      reinterpret_cast<hip::GraphMemcpyNodeFromSymbol*>(clonedNode)->GetMemcpyKind();
  if (oldkind != kind) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  constexpr bool kCheckDeviceIsSame = true;
  hipError_t status = reinterpret_cast<hip::GraphMemcpyNodeFromSymbol*>(clonedNode)
                          ->SetParams(dst, symbol, count, offset, kind, kCheckDeviceIsSame);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    status = reinterpret_cast<hip::GraphExec*>(hGraphExec)
                 ->UpdateAQLPacket(reinterpret_cast<hip::GraphKernelNode*>(clonedNode));
  }
  HIP_RETURN(status);
}

hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphAddMemcpyNodeToSymbol, pGraphNode, graph, pDependencies, numDependencies,
               symbol, src, count, offset, kind);
  if (pGraphNode == nullptr || graph == nullptr || src == nullptr || count == 0 ||
      !hip::Graph::isGraphValid(reinterpret_cast<hip::Graph*>(graph)) ||
      (pDependencies == nullptr && numDependencies > 0)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  size_t sym_size = 0;
  hipDeviceptr_t device_ptr = nullptr;
  hipError_t status = ihipMemcpySymbol_validate(symbol, count, offset, sym_size, device_ptr);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  hip::GraphNode* node = new hip::GraphMemcpyNodeToSymbol(symbol, src, count, offset, kind);
  if (node == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                            reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                            numDependencies, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphMemcpyNodeSetParamsToSymbol, symbol, src, count, offset, kind);
  if (symbol == nullptr) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) || src == nullptr ||
      count == 0 || symbol == src) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(reinterpret_cast<hip::GraphMemcpyNodeToSymbol*>(node)->SetParams(symbol, src, count,
                                                                              offset, kind));
}


hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset,
                                                   hipMemcpyKind kind) {
  HIP_INIT_API(hipGraphExecMemcpyNodeSetParamsToSymbol, hGraphExec, node, symbol, src, count,
               offset, kind);
  if (symbol == nullptr) {
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (hGraphExec == nullptr || src == nullptr || !hip::GraphNode::isNodeValid(n) || count == 0 ||
      src == symbol) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipMemcpyKind oldkind =
      reinterpret_cast<hip::GraphMemcpyNodeToSymbol*>(clonedNode)->GetMemcpyKind();
  if (oldkind != kind) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  constexpr bool kCheckDeviceIsSame = true;
  hipError_t status = reinterpret_cast<hip::GraphMemcpyNodeToSymbol*>(clonedNode)
                          ->SetParams(symbol, src, count, offset, kind, kCheckDeviceIsSame);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    status = reinterpret_cast<hip::GraphExec*>(hGraphExec)
                 ->UpdateAQLPacket(reinterpret_cast<hip::GraphKernelNode*>(clonedNode));
  }
  HIP_RETURN(status);
}

hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event) {
  HIP_INIT_API(hipGraphAddEventRecordNode, pGraphNode, graph, pDependencies, numDependencies,
               event);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node = new hip::GraphEventRecordNode(event);
  hipError_t status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                       reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                       numDependencies, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  HIP_INIT_API(hipGraphEventRecordNodeGetEvent, node, event_out);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (!hip::GraphNode::isNodeValid(n) || event_out == nullptr ||
      n->GetType() != hipGraphNodeTypeEventRecord) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphEventRecordNode*>(n)->GetParams(event_out);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  HIP_INIT_API(hipGraphEventRecordNodeSetEvent, node, event);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);

  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(n)) || event == nullptr ||
      n->GetType() != hipGraphNodeTypeEventRecord) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphEventRecordNode*>(n)->SetParams(event));
}

hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event) {
  HIP_INIT_API(hipGraphExecEventRecordNodeSetEvent, hGraphExec, hNode, event);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  if (hGraphExec == nullptr || n == nullptr || event == nullptr ||
      n->GetType() != hipGraphNodeTypeEventRecord) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphEventRecordNode*>(clonedNode)->SetParams(event));
}

hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event) {
  HIP_INIT_API(hipGraphAddEventWaitNode, pGraphNode, graph, pDependencies, numDependencies, event);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || event == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node = new hip::GraphEventWaitNode(event);
  hipError_t status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                       reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                       numDependencies, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out) {
  HIP_INIT_API(hipGraphEventWaitNodeGetEvent, node, event_out);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);

  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(n)) || event_out == nullptr ||
      n->GetType() != hipGraphNodeTypeWaitEvent) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphEventWaitNode*>(n)->GetParams(event_out);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event) {
  HIP_INIT_API(hipGraphEventWaitNodeSetEvent, node, event);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);

  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(n)) || event == nullptr ||
      n->GetType() != hipGraphNodeTypeWaitEvent) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphEventWaitNode*>(n)->SetParams(event));
}

hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event) {
  HIP_INIT_API(hipGraphExecEventWaitNodeSetEvent, hGraphExec, hNode, event);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);

  if (hGraphExec == nullptr || hNode == nullptr || event == nullptr ||
      (n->GetType() != hipGraphNodeTypeWaitEvent)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphEventWaitNode*>(clonedNode)->SetParams(event));
}

hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphAddHostNode, pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
  if (pGraphNode == nullptr || graph == nullptr || pNodeParams == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pNodeParams->fn == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::GraphNode* node = new hip::GraphHostNode(pNodeParams);
  hipError_t status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                       reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                       numDependencies, false);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphHostNodeGetParams, node, pNodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node)) ||
      pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphHostNode*>(node)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphHostNodeSetParams, node, pNodeParams);
  if (pNodeParams == nullptr || pNodeParams->fn == nullptr ||
      !hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(node))) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphHostNode*>(node)->SetParams(pNodeParams));
}

hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphExecHostNodeSetParams, hGraphExec, node, pNodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (hGraphExec == nullptr || pNodeParams == nullptr || pNodeParams->fn == nullptr ||
      !hip::GraphNode::isNodeValid(n) || n->GetType() != hipGraphNodeTypeHost) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphHostNode*>(clonedNode)->SetParams(pNodeParams));
}

hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out) {
  HIP_INIT_API(hipGraphExecUpdate, hGraphExec, hGraph, hErrorNode_out, updateResult_out);
  // parameter check
  if (hGraphExec == nullptr || hGraph == nullptr || hErrorNode_out == nullptr ||
      updateResult_out == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  std::vector<hip::GraphNode*> newGraphNodes;
  reinterpret_cast<hip::Graph*>(hGraph)->TopologicalOrder(newGraphNodes);
  std::vector<hip::GraphNode*>& oldGraphExecNodes =
      reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetNodes();
  if (newGraphNodes.size() != oldGraphExecNodes.size()) {
    *updateResult_out = hipGraphExecUpdateErrorTopologyChanged;
    *hErrorNode_out = nullptr;
    HIP_RETURN(hipErrorGraphExecUpdateFailure);
  }

  for (std::vector<hip::GraphNode*>::size_type i = 0; i != newGraphNodes.size(); i++) {
    // Checks if all the node types are same before updating
    if (newGraphNodes[i]->GetType() == oldGraphExecNodes[i]->GetType()) {
      if (newGraphNodes[i]->GetType() != hipGraphNodeTypeHost &&
          newGraphNodes[i]->GetType() != hipGraphNodeTypeEmpty) {
        if (newGraphNodes[i]->GetParentGraph()->Device() !=
            oldGraphExecNodes[i]->GetParentGraph()->Device()) {
          *updateResult_out = hipGraphExecUpdateErrorUnsupportedFunctionChange;
          *hErrorNode_out = reinterpret_cast<hipGraphNode_t>(newGraphNodes[i]);
          return hipErrorGraphExecUpdateFailure;
        }
      }

      if (newGraphNodes[i]->GetType() == hipGraphNodeTypeMemcpy) {
        // Checks if the memcpy node's parameters are same
        const hip::GraphMemcpyNode* newMemcpyNode =
            static_cast<hip::GraphMemcpyNode const*>(newGraphNodes[i]);
        const hip::GraphMemcpyNode* oldMemcpyNode =
            static_cast<hip::GraphMemcpyNode const*>(oldGraphExecNodes[i]);
        hipMemcpyKind newKind, oldKind;
        newKind = newMemcpyNode->GetMemcpyKind();
        oldKind = oldMemcpyNode->GetMemcpyKind();
        if (newKind != oldKind) {
          *hErrorNode_out = reinterpret_cast<hipGraphNode_t>(newGraphNodes[i]);
          *updateResult_out = hipGraphExecUpdateErrorParametersChanged;
          HIP_RETURN(hipErrorGraphExecUpdateFailure);
        }
      }
      // Checks if all the node's dependencies are same
      const std::vector<hip::GraphNode*>& newGraphDependencies =
          newGraphNodes[i]->GetDependencies();
      const std::vector<hip::GraphNode*>& oldGraphDependencies =
          oldGraphExecNodes[i]->GetDependencies();
      if (newGraphDependencies.size() != oldGraphDependencies.size()) {
        *hErrorNode_out = reinterpret_cast<hipGraphNode_t>(newGraphNodes[i]);
        *updateResult_out = hipGraphExecUpdateErrorTopologyChanged;
        HIP_RETURN(hipErrorGraphExecUpdateFailure);
      }

      hipError_t status = oldGraphExecNodes[i]->SetParams(newGraphNodes[i]);
      if (status != hipSuccess) {
        *hErrorNode_out = reinterpret_cast<hipGraphNode_t>(newGraphNodes[i]);
        if (status == hipErrorInvalidDeviceFunction) {
          *updateResult_out = hipGraphExecUpdateErrorUnsupportedFunctionChange;
        } else if (status == hipErrorInvalidValue || status == hipErrorInvalidDevicePointer) {
          *updateResult_out = hipGraphExecUpdateErrorParametersChanged;
        } else {
          *updateResult_out = hipGraphExecUpdateErrorNotSupported;
        }
        HIP_RETURN(hipErrorGraphExecUpdateFailure);
      } else if (DEBUG_CLR_GRAPH_PACKET_CAPTURE && newGraphNodes[i]->GraphCaptureEnabled()) {
        status =
            reinterpret_cast<hip::GraphExec*>(hGraphExec)
                ->UpdateAQLPacket(reinterpret_cast<hip::GraphKernelNode*>(oldGraphExecNodes[i]));
      }
    } else {
      *hErrorNode_out = reinterpret_cast<hipGraphNode_t>(newGraphNodes[i]);
      *updateResult_out = hipGraphExecUpdateErrorNodeTypeChanged;
      HIP_RETURN(hipErrorGraphExecUpdateFailure);
    }
  }
  *updateResult_out = hipGraphExecUpdateSuccess;
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   hipMemAllocNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphAddMemAllocNode, pGraphNode, graph, pDependencies, numDependencies,
               pNodeParams);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pNodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (pNodeParams->bytesize == 0 ||
      pNodeParams->poolProps.allocType != hipMemAllocationTypePinned ||
      pNodeParams->poolProps.location.type != hipMemLocationTypeDevice) {
    pNodeParams->dptr = nullptr;
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (pNodeParams->poolProps.location.type == hipMemLocationTypeDevice) {
    if (pNodeParams->poolProps.location.id < 0 ||
        pNodeParams->poolProps.location.id >= g_devices.size()) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }
  // Clear the pointer to allocated memory because it may contain stale/uninitialized data
  pNodeParams->dptr = nullptr;
  auto mem_alloc_node = new hip::GraphMemAllocNode(pNodeParams);
  hip::GraphNode* node = mem_alloc_node;
  auto hgraph = reinterpret_cast<hip::Graph*>(graph);
  auto status = ihipGraphAddNode(
      node, hgraph, reinterpret_cast<hip::GraphNode* const*>(pDependencies), numDependencies);
  // The address must be provided during the node creation time
  pNodeParams->dptr =
      (HIP_MEM_POOL_USE_VM) ? mem_alloc_node->ReserveAddress() : mem_alloc_node->Execute();
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  amd::ScopedLock lock(hip::Graph::graphSetLock_);
  hgraph->memAllocNodePtrs_.insert(pNodeParams->dptr);
  HIP_RETURN(status);
}

// ================================================================================================
hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphMemAllocNodeGetParams, node, pNodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (node == nullptr || pNodeParams == nullptr ||
      !hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(n)) ||
      n->GetType() != hipGraphNodeTypeMemAlloc) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphMemAllocNode*>(n)->GetParams(pNodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t ihipGraphAddMemFreeNode(hip::GraphNode** graphNode, hip::Graph* graph,
                                   hip::GraphNode* const* pDependencies, size_t numDependencies,
                                   void* dptr) {
  // Is memory passed to be free'd valid
  size_t offset = 0;
  auto memory = getMemoryObject(dptr, offset);
  if (memory == nullptr) {
    if (HIP_MEM_POOL_USE_VM) {
      // When VM is on the address must be valid and may point to a VA object
      memory = amd::MemObjMap::FindVirtualMemObj(dptr);
    }
    if (memory == nullptr) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }

  auto mem_free_node = new hip::GraphMemFreeNode(dptr);
  *graphNode = mem_free_node;
  auto status = ihipGraphAddNode(*graphNode, graph, pDependencies, numDependencies);
  HIP_RETURN(status);
}
// ================================================================================================
hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  void* dev_ptr) {
  HIP_INIT_API(hipGraphAddMemFreeNode, pGraphNode, graph, pDependencies, numDependencies, dev_ptr);
  if (pGraphNode == nullptr || graph == nullptr ||
      ((numDependencies > 0 && pDependencies == nullptr) ||
       (pDependencies != nullptr && numDependencies == 0)) ||
      dev_ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // memAllocNodePtrs_ stores only local to graph alloc dptrs whose free node is not added.
  // and so we need to traverse all graphs of graphSet_ and below cases are handled.
  // 1) Free node cannot be added twice to the same graph
  // 2) Free node if it part of another graph cannot be added to this graph
  hip::GraphNode* pNode;
  bool bGraphFound = false;
  {
    amd::ScopedLock lock(hip::Graph::graphSetLock_);
    for (auto itGraph : hip::Graph::graphSet_) {
      std::unordered_set<void*>::iterator itDevPtr = itGraph->memAllocNodePtrs_.find(dev_ptr);
      if (itDevPtr != itGraph->memAllocNodePtrs_.end()) {
        bGraphFound = true;
        itGraph->memAllocNodePtrs_.erase(itDevPtr);
        break;
      }
    }
  }
  if (bGraphFound == false) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  auto status = ihipGraphAddMemFreeNode(&pNode, reinterpret_cast<hip::Graph*>(graph),
                                        reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                        numDependencies, dev_ptr);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(pNode);
  HIP_RETURN(status);
}

// ================================================================================================
hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr) {
  HIP_INIT_API(hipGraphMemFreeNodeGetParams, node, dev_ptr);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (node == nullptr || dev_ptr == nullptr || !hip::GraphNode::isNodeValid(n) ||
      n->GetType() != hipGraphNodeTypeMemFree) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphMemFreeNode*>(n)->GetParams(reinterpret_cast<void**>(dev_ptr));
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  HIP_INIT_API(hipDeviceGetGraphMemAttribute, device, attr, value);
  if ((static_cast<size_t>(device) >= g_devices.size()) || device < 0) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  if (value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipError_t result = hipErrorInvalidValue;
  switch (attr) {
    case hipGraphMemAttrUsedMemCurrent:
      result = g_devices[device]->GetGraphMemoryPool()->GetAttribute(hipMemPoolAttrUsedMemCurrent,
                                                                     value);
      break;
    case hipGraphMemAttrUsedMemHigh:
      result =
          g_devices[device]->GetGraphMemoryPool()->GetAttribute(hipMemPoolAttrUsedMemHigh, value);
      break;
    case hipGraphMemAttrReservedMemCurrent:
      result = g_devices[device]->GetGraphMemoryPool()->GetAttribute(
          hipMemPoolAttrReservedMemCurrent, value);
      break;
    case hipGraphMemAttrReservedMemHigh:
      result = g_devices[device]->GetGraphMemoryPool()->GetAttribute(hipMemPoolAttrReservedMemHigh,
                                                                     value);
      break;
    default:
      break;
  }
  HIP_RETURN(result);
}

// ================================================================================================
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value) {
  HIP_INIT_API(hipDeviceSetGraphMemAttribute, device, attr, value);
  if ((static_cast<size_t>(device) >= g_devices.size()) || device < 0) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  if (value == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipError_t result = hipErrorInvalidValue;
  switch (attr) {
    case hipGraphMemAttrUsedMemHigh:
      result =
          g_devices[device]->GetGraphMemoryPool()->SetAttribute(hipMemPoolAttrUsedMemHigh, value);
      break;
    case hipGraphMemAttrReservedMemHigh:
      result = g_devices[device]->GetGraphMemoryPool()->SetAttribute(hipMemPoolAttrReservedMemHigh,
                                                                     value);
      break;
    default:
      break;
  }
  HIP_RETURN(result);
}

// ================================================================================================
hipError_t hipDeviceGraphMemTrim(int device) {
  HIP_INIT_API(hipDeviceGraphMemTrim, device);
  if ((static_cast<size_t>(device) >= g_devices.size()) || device < 0) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  g_devices[device]->GetGraphMemoryPool()->TrimTo(0);
  HIP_RETURN(hipSuccess);
}

// ================================================================================================
hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
                               unsigned int initialRefcount, unsigned int flags) {
  HIP_INIT_API(hipUserObjectCreate, object_out, ptr, destroy, initialRefcount, flags);
  if (object_out == nullptr || flags != hipUserObjectNoDestructorSync || initialRefcount == 0 ||
      destroy == nullptr || initialRefcount > INT_MAX) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::UserObject* object = new hip::UserObject(destroy, ptr, flags);
  //! Creating object adds one reference.
  if (initialRefcount > 1) {
    object->increaseRefCount(static_cast<const unsigned int>(initialRefcount - 1));
  }
  *object_out = reinterpret_cast<hipUserObject_t>(object);
  HIP_RETURN(hipSuccess);
}

hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count) {
  HIP_INIT_API(hipUserObjectRelease, object, count);
  if (object == nullptr || count == 0 || count > INT_MAX) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::UserObject* userObject = reinterpret_cast<hip::UserObject*>(object);
  if (userObject->referenceCount() < count || !hip::UserObject::isUserObjvalid(userObject)) {
    HIP_RETURN(hipSuccess);
  }
  //! If all the counts are gone not longer need the obj in the list
  //! If user object is about to be deleted, It's reference needs to be
  //! removed from graph's user object list.
  if (userObject->referenceCount() == count) {
    hip::UserObject::removeUSerObj(userObject);
    for (auto& g : userObject->owning_graphs_) {
      g->RemoveUserObjGraph(userObject);
    }
  }
  userObject->decreaseRefCount(count);
  HIP_RETURN(hipSuccess);
}

hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count) {
  HIP_INIT_API(hipUserObjectRetain, object, count);
  if (object == nullptr || count == 0 || count > INT_MAX) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::UserObject* userObject = reinterpret_cast<hip::UserObject*>(object);
  if (!hip::UserObject::isUserObjvalid(userObject)) {
    HIP_RETURN(hipSuccess);
  }
  userObject->increaseRefCount(count);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count,
                                    unsigned int flags) {
  HIP_INIT_API(hipGraphRetainUserObject, graph, object, count, flags);
  hipError_t status = hipSuccess;
  if (graph == nullptr || object == nullptr || count == 0 || count > INT_MAX ||
      (flags != 0 && flags != hipGraphUserObjectMove)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::UserObject* userObject = reinterpret_cast<hip::UserObject*>(object);
  hip::Graph* g = reinterpret_cast<hip::Graph*>(graph);
  if (!hip::UserObject::isUserObjvalid(userObject) && !g->isUserObjGraphValid(userObject)) {
    HIP_RETURN(hipSuccess);
  }
  if (flags != hipGraphUserObjectMove) {
    status = hip::hipUserObjectRetain(object, count);
    if (status != hipSuccess) {
      HIP_RETURN(status);
    }
  }
  g->addUserObjGraph(userObject);
  userObject->owning_graphs_.insert(g);
  g->IncrementGraphUserObjRefCount(userObject, count);
  HIP_RETURN(status);
}

hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count) {
  HIP_INIT_API(hipGraphReleaseUserObject, graph, object, count);
  if (graph == nullptr || object == nullptr || count == 0 || count > INT_MAX) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::UserObject* userObject = reinterpret_cast<hip::UserObject*>(object);
  hip::Graph* g = reinterpret_cast<hip::Graph*>(graph);
  if (!g->isUserObjGraphValid(userObject) || userObject->referenceCount() < count) {
    HIP_RETURN(hipSuccess);
  }
  g->DecrementGraphUserObjRefCount(userObject, count);
  //! Obj is being destroyed
  unsigned int releaseCount =
      (userObject->referenceCount() < count) ? userObject->referenceCount() : count;
  if (userObject->referenceCount() == releaseCount) {
    g->RemoveUserObjGraph(userObject);
    //! If user object is about to be deleted, Its owner needs to be updated.
    userObject->owning_graphs_.erase(g);
  }
  hipError_t status = hip::hipUserObjectRelease(object, count);
  HIP_RETURN(status);
}

hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst) {
  HIP_INIT_API(hipGraphKernelNodeCopyAttributes, hSrc, hDst);
  if (hSrc == nullptr || hDst == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphKernelNode*>(hDst)->CopyAttr(
      reinterpret_cast<hip::GraphKernelNode*>(hSrc)));
}

hipError_t ihipGraphDebugDotPrint(hip::Graph* graph, const char* path, unsigned int flags) {
  std::ofstream fout;
  fout.open(path, std::ios::out);
  if (fout.fail()) {
    LogPrintfError("[hipGraph] Error during opening of file : %s", path);
    return hipErrorOperatingSystem;
  }
  fout << "digraph dot {" << std::endl;
  hip::Graph* g = reinterpret_cast<hip::Graph*>(graph);
  g->GenerateDOT(fout, (hipGraphDebugDotFlags)flags);
  fout << "}" << std::endl;
  fout.close();
  return hipSuccess;
}

hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags) {
  HIP_INIT_API(hipGraphDebugDotPrint, graph, path, flags);
  if (graph == nullptr || path == nullptr) {
    return hipErrorInvalidValue;
  }
  hip::Graph* hip_graph = reinterpret_cast<hip::Graph*>(graph);
  HIP_RETURN(ihipGraphDebugDotPrint(hip_graph, path, flags));
}

hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int isEnabled) {
  HIP_INIT_API(hipGraphNodeSetEnabled, hGraphExec, hNode, isEnabled);
  hip::GraphExec* graphExec = reinterpret_cast<hip::GraphExec*>(hGraphExec);
  hip::GraphNode* node = reinterpret_cast<hip::GraphNode*>(hNode);
  if (hGraphExec == nullptr || hNode == nullptr || !hip::GraphExec::isGraphExecValid(graphExec) ||
      !hip::GraphNode::isNodeValid(node)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = graphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!(node->GetType() == hipGraphNodeTypeKernel || node->GetType() == hipGraphNodeTypeMemcpy ||
        node->GetType() == hipGraphNodeTypeMemset)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  clonedNode->SetEnabled(isEnabled);
  // Update packet batches when node is enabled/disabled
  hipError_t status = graphExec->UpdatePacketBatchesForNodeEnableDisable(clonedNode, isEnabled != 0);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int* isEnabled) {
  HIP_INIT_API(hipGraphNodeGetEnabled, hGraphExec, hNode, isEnabled);
  hip::GraphExec* graphExec = reinterpret_cast<hip::GraphExec*>(hGraphExec);
  hip::GraphNode* node = reinterpret_cast<hip::GraphNode*>(hNode);

  if (hGraphExec == nullptr || hNode == nullptr || isEnabled == nullptr ||
      !hip::GraphExec::isGraphExecValid(graphExec) || !hip::GraphNode::isNodeValid(node)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = graphExec->GetClonedNode(node);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!(node->GetType() == hipGraphNodeTypeKernel || node->GetType() == hipGraphNodeTypeMemcpy ||
        node->GetType() == hipGraphNodeTypeMemset)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  *isEnabled = clonedNode->GetEnabled();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream) {
  HIP_INIT_API(hipGraphUpload, graphExec, stream);
  // TODO: stream is known before launch, do preperatory work with graph optimizations. pre-allocate
  // memory for memAlloc nodes if any when support is added with mempool feature
  hipError_t status = ihipGraphUpload(graphExec, stream);
  HIP_RETURN(status);
}

hipError_t hipGraphAddNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                           const hipGraphNode_t* pDependencies, size_t numDependencies,
                           hipGraphNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphAddNode, pGraphNode, graph, pDependencies, numDependencies, nodeParams);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipGraphNodeType nodeType = nodeParams->type;
  hip::GraphNode* node;
  hipError_t status = hipSuccess;
  hip::GraphMemAllocNode* mem_alloc_node;
  amd::Memory* memory;
  size_t offset;
  hipMemAllocNodeParams params;

  switch (nodeType) {
    case hipGraphNodeTypeKernel:
      status = ihipGraphAddKernelNode(&node, reinterpret_cast<hip::Graph*>(graph),
                                      reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                      numDependencies, &nodeParams->kernel, nullptr, false);
      break;
    case hipGraphNodeTypeMemcpy:
      status = ihipGraphAddMemcpyNode(&node, reinterpret_cast<hip::Graph*>(graph),
                                      reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                      numDependencies, &nodeParams->memcpy.copyParams, false);
      break;
    case hipGraphNodeTypeMemset:
      status = ihipGraphAddMemsetNode(&node, reinterpret_cast<hip::Graph*>(graph),
                                      reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                      numDependencies, &nodeParams->memset, false);
      break;
    case hipGraphNodeTypeHost:
      if (nodeParams->host.fn == nullptr) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      node = new hip::GraphHostNode(&nodeParams->host);
      status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                numDependencies, false);
      break;
    case hipGraphNodeTypeGraph:
      if (nodeParams->graph.graph == nullptr) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      node = new hip::ChildGraphNode(reinterpret_cast<hip::Graph*>(nodeParams->graph.graph));
      status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                numDependencies, false);
      break;
    case hipGraphNodeTypeWaitEvent:
      if (nodeParams->eventWait.event == nullptr) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      node = new hip::GraphEventWaitNode(nodeParams->eventWait.event);
      status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                numDependencies, false);
      break;
    case hipGraphNodeTypeEventRecord:
      if (nodeParams->eventRecord.event == nullptr) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      node = new hip::GraphEventRecordNode(nodeParams->eventRecord.event);
      status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                numDependencies, false);
      break;
    case hipGraphNodeTypeExtSemaphoreSignal:
      status = hipErrorNotSupported;
      // to be added.
      break;
    case hipGraphNodeTypeExtSemaphoreWait:
      status = hipErrorNotSupported;
      // to be added.
      break;
    case hipGraphNodeTypeMemAlloc:
      params = nodeParams->alloc;
      if (params.bytesize == 0 || params.poolProps.allocType != hipMemAllocationTypePinned ||
          params.poolProps.location.type != hipMemLocationTypeDevice) {
        params.dptr = nullptr;
        HIP_RETURN(hipErrorInvalidValue);
      }
      if (params.poolProps.location.type == hipMemLocationTypeDevice) {
        if (params.poolProps.location.id < 0 || params.poolProps.location.id >= g_devices.size()) {
          HIP_RETURN(hipErrorInvalidValue);
        }
      }
      // Clear the pointer to allocated memory because it may contain stale/uninitialized data
      params.dptr = nullptr;
      mem_alloc_node = new hip::GraphMemAllocNode(&params);
      node = mem_alloc_node;
      status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                numDependencies);
      // The address must be provided during the node creation time
      nodeParams->alloc.dptr =
          (HIP_MEM_POOL_USE_VM) ? mem_alloc_node->ReserveAddress() : mem_alloc_node->Execute();
      break;
    case hipGraphNodeTypeMemFree:
      if (nodeParams->free.dptr == nullptr) {
        HIP_RETURN(hipErrorInvalidValue);
      }
      // Is memory passed to be free'd valid
      offset = 0;
      memory = getMemoryObject(nodeParams->free.dptr, offset);
      if (memory == nullptr) {
        if (HIP_MEM_POOL_USE_VM) {
          // When VM is on the address must be valid and may point to a VA object
          memory = amd::MemObjMap::FindVirtualMemObj(nodeParams->free.dptr);
        }
        if (memory == nullptr) {
          HIP_RETURN(hipErrorInvalidValue);
        }
      }
      node = new hip::GraphMemFreeNode(nodeParams->free.dptr);
      status = ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                                reinterpret_cast<hip::GraphNode* const*>(pDependencies),
                                numDependencies);
      break;
    default:
      status = hipErrorInvalidValue;
      break;
  }
  *pGraphNode = reinterpret_cast<hipGraphNode*>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphAddExternalSemaphoresSignalNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphAddExternalSemaphoresSignalNode, pGraphNode, graph, pDependencies,
               numDependencies, nodeParams);
  hip::GraphNode* node = new hip::hipGraphExternalSemSignalNode(nodeParams);
  hipError_t status =
      ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                       reinterpret_cast<hip::GraphNode* const*>(pDependencies), numDependencies);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphAddExternalSemaphoresWaitNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphAddExternalSemaphoresWaitNode, pGraphNode, graph, pDependencies,
               numDependencies, nodeParams);
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* node = new hip::hipGraphExternalSemWaitNode(nodeParams);
  hipError_t status =
      ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(graph),
                       reinterpret_cast<hip::GraphNode* const*>(pDependencies), numDependencies);
  *pGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphExternalSemaphoresSignalNodeSetParams, hNode, nodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  if (!hip::GraphNode::isNodeValid(n) || nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::hipGraphExternalSemSignalNode*>(n)->SetParams(nodeParams));
}

hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphExternalSemaphoresWaitNodeSetParams, hNode, nodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  if (!hip::GraphNode::isNodeValid(n) || nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::hipGraphExternalSemWaitNode*>(n)->SetParams(nodeParams));
}

hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out) {
  HIP_INIT_API(hipGraphExternalSemaphoresSignalNodeGetParams, hNode, params_out);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  if (!hip::GraphNode::isNodeValid(n) || params_out == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::hipGraphExternalSemSignalNode*>(n)->GetParams(params_out);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out) {
  HIP_INIT_API(hipGraphExternalSemaphoresWaitNodeGetParams, hNode, params_out);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  if (!hip::GraphNode::isNodeValid(n) || params_out == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::hipGraphExternalSemWaitNode*>(n)->GetParams(params_out);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreSignalNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphExecExternalSemaphoresSignalNodeSetParams, hGraphExec, hNode, nodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  hip::GraphExec* graphExec = reinterpret_cast<hip::GraphExec*>(hGraphExec);
  if (hGraphExec == nullptr || hNode == nullptr || !hip::GraphExec::isGraphExecValid(graphExec) ||
      !hip::GraphNode::isNodeValid(n) || nodeParams == nullptr ||
      n->GetType() != hipGraphNodeTypeExtSemaphoreSignal) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = graphExec->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(
      reinterpret_cast<hip::hipGraphExternalSemSignalNode*>(clonedNode)->SetParams(nodeParams));
}

hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreWaitNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphExecExternalSemaphoresWaitNodeSetParams, hGraphExec, hNode, nodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  hip::GraphExec* graphExec = reinterpret_cast<hip::GraphExec*>(hGraphExec);
  if (hGraphExec == nullptr || hNode == nullptr || !hip::GraphExec::isGraphExecValid(graphExec) ||
      !hip::GraphNode::isNodeValid(n) || nodeParams == nullptr ||
      n->GetType() != hipGraphNodeTypeExtSemaphoreWait) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = graphExec->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(
      reinterpret_cast<hip::hipGraphExternalSemWaitNode*>(clonedNode)->SetParams(nodeParams));
}

hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     hipDeviceptr_t dptr) {
  HIP_INIT_API(hipDrvGraphAddMemFreeNode, phGraphNode, hGraph, dependencies, numDependencies, dptr);
  if (phGraphNode == nullptr || hGraph == nullptr ||
      ((numDependencies > 0 && dependencies == nullptr) ||
       (dependencies != nullptr && numDependencies == 0)) ||
      dptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Is memory passed to be free'd valid
  size_t offset = 0;
  auto memory = getMemoryObject(dptr, offset);
  if (memory == nullptr) {
    if (HIP_MEM_POOL_USE_VM) {
      // When VM is on the address must be valid and may point to a VA object
      memory = amd::MemObjMap::FindVirtualMemObj(dptr);
    }
    if (memory == nullptr) {
      HIP_RETURN(hipErrorInvalidValue);
    }
  }
  hip::GraphNode* pNode;
  auto status = ihipGraphAddMemFreeNode(&pNode, reinterpret_cast<hip::Graph*>(hGraph),
                                        reinterpret_cast<hip::GraphNode* const*>(dependencies),
                                        numDependencies, dptr);
  *phGraphNode = reinterpret_cast<hipGraphNode_t>(pNode);
  HIP_RETURN(status);
}

hipError_t hipDrvGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const HIP_MEMCPY3D* copyParams, hipCtx_t ctx) {
  HIP_INIT_API(hipDrvGraphExecMemcpyNodeSetParams, hGraphExec, hNode, copyParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  if (hGraphExec == nullptr || !hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(n))) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (ihipDrvMemcpy3D_validate(copyParams) != hipSuccess) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Check if pNodeParams passed is a empty struct
  if (((copyParams->srcArray == 0) && (copyParams->srcHost == nullptr) &&
       (copyParams->srcDevice == nullptr)) ||
      ((copyParams->dstArray == 0) && (copyParams->dstHost == nullptr) &&
       (copyParams->dstDevice == nullptr))) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphDrvMemcpyNode*>(clonedNode)->SetParams(copyParams));
}

hipError_t hipDrvGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const hipMemsetParams* memsetParams, hipCtx_t ctx) {
  HIP_INIT_API(hipDrvGraphExecMemsetNodeSetParams, hGraphExec, hNode, memsetParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);

  if (hGraphExec == nullptr || !hip::GraphNode::isNodeValid(n) || memsetParams == nullptr ||
      memsetParams->dst == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipMemsetParams pmemsetParams;
  pmemsetParams.dst = reinterpret_cast<void*>(memsetParams->dst);
  pmemsetParams.elementSize = memsetParams->elementSize;
  pmemsetParams.height = memsetParams->height;
  pmemsetParams.pitch = memsetParams->pitch;
  pmemsetParams.value = memsetParams->value;
  pmemsetParams.width = memsetParams->width;
  if (ihipGraphMemsetParams_validate(&pmemsetParams) != hipSuccess) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(hGraphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipError_t status =
      reinterpret_cast<hip::GraphMemsetNode*>(clonedNode)->SetParams(memsetParams, true);
  if (status != hipSuccess) {
    HIP_RETURN(status);
  }
  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    status = reinterpret_cast<hip::GraphExec*>(hGraphExec)->UpdateAQLPacket(clonedNode);
  }
  HIP_RETURN(status);
}

hipError_t hipGraphExecGetFlags(hipGraphExec_t graphExec, unsigned long long* flags) {
  HIP_INIT_API(hipGraphExecGetFlags, graphExec, flags);
  if (graphExec == nullptr || flags == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphExec* pgraphExec = reinterpret_cast<hip::GraphExec*>(graphExec);
  *flags = pgraphExec->GetFlags();
  HIP_RETURN(hipSuccess);
}

hipError_t ihipGraphNodeSetParams(hip::GraphNode* n, hipGraphNodeParams* nodeParams,
                                  bool exec = false) {
  hipGraphNodeType nodeType = nodeParams->type;
  std::vector<hip::GraphNode*> childGraphNodes1;
  std::vector<hip::GraphNode*> childGraphNodes2;
  hip::Graph* cg;
  hipError_t status = hipSuccess;
  switch (nodeType) {
    case hipGraphNodeTypeKernel:
      status = reinterpret_cast<hip::GraphKernelNode*>(n)->SetParams(&nodeParams->kernel);
      break;
    case hipGraphNodeTypeMemcpy:
      if (exec) {  // this validation is only required for ExecNodeSetParams
        hipMemcpyKind oldkind = reinterpret_cast<hip::GraphMemcpyNode*>(n)->GetMemcpyKind();
        hipMemcpyKind newkind = nodeParams->memcpy.copyParams.kind;
        if (oldkind != newkind) {
          status = hipErrorInvalidValue;
          break;
        }
      }
      status =
          reinterpret_cast<hip::GraphMemcpyNode*>(n)->SetParams(&nodeParams->memcpy.copyParams);
      break;
    case hipGraphNodeTypeMemset:
      status = reinterpret_cast<hip::GraphMemsetNode*>(n)->SetParams(&nodeParams->memset);
      break;
    case hipGraphNodeTypeHost:
      if (nodeParams->host.fn == nullptr || nodeParams->host.userData == nullptr) {
        status = hipErrorInvalidValue;
        break;
      }
      status = reinterpret_cast<hip::GraphHostNode*>(n)->SetParams(&nodeParams->host);
      break;
    case hipGraphNodeTypeGraph:
      cg = reinterpret_cast<hip::Graph*>(nodeParams->graph.graph);
      status = validateChildGraphNodeSetParams(n, cg, exec);
      if (status != hipSuccess) {
        break;
      }
      status = reinterpret_cast<hip::ChildGraphNode*>(n)->SetParams(
          reinterpret_cast<hip::Graph*>(nodeParams->graph.graph));
      break;
    case hipGraphNodeTypeWaitEvent:
      if (nodeParams->eventWait.event == nullptr) {
        status = hipErrorInvalidValue;
        break;
      }
      status =
          reinterpret_cast<hip::GraphEventWaitNode*>(n)->SetParams(nodeParams->eventWait.event);
      break;
    case hipGraphNodeTypeEventRecord:
      if (nodeParams->eventRecord.event == nullptr) {
        status = hipErrorInvalidValue;
        break;
      }
      status =
          reinterpret_cast<hip::GraphEventRecordNode*>(n)->SetParams(nodeParams->eventRecord.event);
      break;
    case hipGraphNodeTypeExtSemaphoreSignal:
      status = reinterpret_cast<hip::hipGraphExternalSemSignalNode*>(n)->SetParams(
          &nodeParams->extSemSignal);
      break;
    case hipGraphNodeTypeExtSemaphoreWait:
      status = reinterpret_cast<hip::hipGraphExternalSemWaitNode*>(n)->SetParams(
          &nodeParams->extSemWait);
      break;
    case hipGraphNodeTypeMemAlloc:
      status = hipErrorNotSupported;
      break;
    case hipGraphNodeTypeMemFree:
      status = hipErrorNotSupported;
      break;
    default:
      status = hipErrorInvalidValue;
      break;
  }
  HIP_RETURN(status);
}

hipError_t hipGraphNodeSetParams(hipGraphNode_t node, hipGraphNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphNodeSetParams, node, nodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (node == nullptr || nodeParams == nullptr || !hip::GraphNode::isNodeValid(n)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(ihipGraphNodeSetParams(n, nodeParams, false));
}

hipError_t hipGraphExecNodeSetParams(hipGraphExec_t graphExec, hipGraphNode_t node,
                                     hipGraphNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphNodeSetParams, graphExec, node, nodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(node);
  if (node == nullptr || nodeParams == nullptr || graphExec == nullptr ||
      !hip::GraphNode::isNodeValid(n)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphNode*>(
      reinterpret_cast<hip::GraphExec*>(graphExec)->GetClonedNode(n));
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipError_t status = ihipGraphNodeSetParams(clonedNode, nodeParams, true);
  if (status != hipSuccess) {
    return status;
  }

  if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
    status = reinterpret_cast<hip::GraphExec*>(graphExec)->UpdateAQLPacket(clonedNode);
  }
  return status;
}

hipError_t hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode, HIP_MEMCPY3D* nodeParams) {
  HIP_INIT_API(hipDrvGraphMemcpyNodeGetParams, hNode, nodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(hNode)) ||
      nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::GraphDrvMemcpyNode*>(hNode)->GetParams(nodeParams);
  HIP_RETURN(hipSuccess);
}

hipError_t hipDrvGraphMemcpyNodeSetParams(hipGraphNode_t hNode, const HIP_MEMCPY3D* nodeParams) {
  HIP_INIT_API(hipDrvGraphMemcpyNodeSetParams, hNode, nodeParams);
  if (!hip::GraphNode::isNodeValid(reinterpret_cast<hip::GraphNode*>(hNode)) ||
      nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::GraphDrvMemcpyNode*>(hNode)->SetParams(nodeParams));
}

hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     const hipBatchMemOpNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphAddBatchMemOpNode, phGraphNode, hGraph, dependencies, numDependencies,
               nodeParams);
  if (phGraphNode == nullptr || hGraph == nullptr ||
      (numDependencies > 0 && dependencies == nullptr) || nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Check nodeParams fields
  if (nodeParams->count <= 0 || nodeParams->count > 256 || nodeParams->paramArray == nullptr ||
      nodeParams->flags != 0 || nodeParams->ctx == nullptr) {
    return hipErrorInvalidValue;
  }

  hip::GraphNode* node = new hip::hipGraphBatchMemOpNode(nodeParams);
  hipError_t status =
      ihipGraphAddNode(node, reinterpret_cast<hip::Graph*>(hGraph),
                       reinterpret_cast<hip::GraphNode* const*>(dependencies), numDependencies);
  *phGraphNode = reinterpret_cast<hipGraphNode_t>(node);
  HIP_RETURN(status);
}

hipError_t hipGraphBatchMemOpNodeGetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams_out) {
  HIP_INIT_API(hipGraphBatchMemOpNodeGetParams, hNode, nodeParams_out);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  if (!hip::GraphNode::isNodeValid(n) || nodeParams_out == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  reinterpret_cast<hip::hipGraphBatchMemOpNode*>(n)->GetParams(nodeParams_out);
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphBatchMemOpNodeSetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphBatchMemOpNodeSetParams, hNode, nodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  if (!hip::GraphNode::isNodeValid(n) || nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Check nodeParams fields
  if (nodeParams->count <= 0 || nodeParams->count > 256 || nodeParams->paramArray == nullptr ||
      nodeParams->flags != 0 || nodeParams->ctx == nullptr) {
    return hipErrorInvalidValue;
  }
  HIP_RETURN(reinterpret_cast<hip::hipGraphBatchMemOpNode*>(n)->SetParams(nodeParams));
}

hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               const hipBatchMemOpNodeParams* nodeParams) {
  HIP_INIT_API(hipGraphExecBatchMemOpNodeSetParams, hGraphExec, hNode, nodeParams);
  hip::GraphNode* n = reinterpret_cast<hip::GraphNode*>(hNode);
  hip::GraphExec* graphExec = reinterpret_cast<hip::GraphExec*>(hGraphExec);
  if (hGraphExec == nullptr || hNode == nullptr || !hip::GraphExec::isGraphExecValid(graphExec) ||
      !hip::GraphNode::isNodeValid(n) || nodeParams == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  // Check nodeParams fields
  if (nodeParams->count <= 0 || nodeParams->count > 256 || nodeParams->paramArray == nullptr ||
      nodeParams->flags != 0 || nodeParams->ctx == nullptr) {
    return hipErrorInvalidValue;
  }
  hip::GraphNode* clonedNode = reinterpret_cast<hip::GraphExec*>(graphExec)->GetClonedNode(n);
  if (clonedNode == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(reinterpret_cast<hip::hipGraphBatchMemOpNode*>(clonedNode)->SetParams(nodeParams));
}
}  // namespace hip
