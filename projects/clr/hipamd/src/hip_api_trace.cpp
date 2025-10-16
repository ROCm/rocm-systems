/*
    Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
   */

// #include "hip_api_trace.hpp"
#include <hip/amd_detail/hip_api_trace.hpp>

#include "hip_internal.hpp"

#include <cstdint>

#if defined(HIP_ROCPROFILER_REGISTER) && HIP_ROCPROFILER_REGISTER > 0
#include <rocprofiler-register/rocprofiler-register.h>

#define HIP_ROCP_REG_VERSION                                                                       \
  ROCPROFILER_REGISTER_COMPUTE_VERSION_3(HIP_ROCP_REG_VERSION_MAJOR, HIP_ROCP_REG_VERSION_MINOR,   \
                                         HIP_ROCP_REG_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(hip, HIP_ROCP_REG_VERSION)
ROCPROFILER_REGISTER_DEFINE_IMPORT(hip_compiler, HIP_ROCP_REG_VERSION)
ROCPROFILER_REGISTER_DEFINE_IMPORT(hip_tools, HIP_ROCP_REG_VERSION)
#elif !defined(HIP_ROCPROFILER_REGISTER)
#define HIP_ROCPROFILER_REGISTER 0
#endif

namespace hip {
// HIP Internal APIs
hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                                     hipStream_t* stream);
hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                      hipStream_t stream);
void** __hipRegisterFatBinary(const void* data);
void __hipRegisterFunction(void** modules, const void* hostFunction, char* deviceFunction,
                           const char* deviceName, unsigned int threadLimit, uint3* tid, uint3* bid,
                           dim3* blockDim, dim3* gridDim, int* wSize);
void __hipRegisterManagedVar(void* hipModule, void** pointer, void* init_value, const char* name,
                             size_t size, unsigned align);
void __hipRegisterSurface(void** modules, void* var, char* hostVar, char* deviceVar, int type,
                          int ext);
void __hipRegisterTexture(void** modules, void* var, char* hostVar, char* deviceVar, int type,
                          int norm, int ext);
void __hipRegisterVar(void** modules, void* var, char* hostVar, char* deviceVar, int ext,
                      size_t size, int constant, int global);
void __hipUnregisterFatBinary(void** modules);
const char* hipApiName(uint32_t id);
hipError_t hipArray3DCreate(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray);
hipError_t hipArray3DGetDescriptor(HIP_ARRAY3D_DESCRIPTOR* pArrayDescriptor, hipArray_t array);
hipError_t hipArrayCreate(hipArray_t* pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray);
hipError_t hipArrayDestroy(hipArray_t array);
hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray_t array);
hipError_t hipArrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags,
                           hipArray_t array);
hipError_t hipBindTexture(size_t* offset, const textureReference* tex, const void* devPtr,
                          const hipChannelFormatDesc* desc, size_t size);
hipError_t hipBindTexture2D(size_t* offset, const textureReference* tex, const void* devPtr,
                            const hipChannelFormatDesc* desc, size_t width, size_t height,
                            size_t pitch);
hipError_t hipBindTextureToArray(const textureReference* tex, hipArray_const_t array,
                                 const hipChannelFormatDesc* desc);
hipError_t hipBindTextureToMipmappedArray(const textureReference* tex,
                                          hipMipmappedArray_const_t mipmappedArray,
                                          const hipChannelFormatDesc* desc);
hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop);
hipError_t hipChooseDeviceR0000(int* device, const hipDeviceProp_tR0000* properties);
hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);
hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
                                  const hipTextureDesc* pTexDesc,
                                  const struct hipResourceViewDesc* pResViewDesc);
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);
hipError_t hipCtxDestroy(hipCtx_t ctx);
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);
hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, unsigned int* apiVersion);
hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig);
hipError_t hipCtxGetCurrent(hipCtx_t* ctx);
hipError_t hipCtxGetDevice(hipDevice_t* device);
hipError_t hipCtxGetFlags(unsigned int* flags);
hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
hipError_t hipCtxPopCurrent(hipCtx_t* ctx);
hipError_t hipCtxPushCurrent(hipCtx_t ctx);
hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);
hipError_t hipCtxSetCurrent(hipCtx_t ctx);
hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);
hipError_t hipCtxSynchronize(void);
hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem);
hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem);
hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);
hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);
hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);
hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig);
hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device);
hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);
hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit);
hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device);
hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);
hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice,
                                    int dstDevice);
hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig);
hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
hipError_t hipDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements,
                                               const hipChannelFormatDesc* fmtDesc, int device);
hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device);
hipError_t hipDeviceGraphMemTrim(int device);
hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active);
hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);
hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);
hipError_t hipDeviceReset(void);
hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);
hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value);
hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool);
hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);
hipError_t hipDeviceSynchronize(void);
hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);
hipError_t hipDriverGetVersion(int* driverVersion);
hipError_t hipDrvGetErrorName(hipError_t hipError, const char** errorString);
hipError_t hipDrvGetErrorString(hipError_t hipError, const char** errorString);
hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy);
hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy);
hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream);
hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes,
                                      void** data, hipDeviceptr_t ptr);
hipError_t hipEventCreate(hipEvent_t* event);
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
hipError_t hipEventDestroy(hipEvent_t event);
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
hipError_t hipEventQuery(hipEvent_t event);
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
hipError_t hipEventSynchronize(hipEvent_t event);
hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype,
                                        uint32_t* hopcount);
hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                              void** args, size_t sharedMemBytes, hipStream_t stream,
                              hipEvent_t startEvent, hipEvent_t stopEvent, int flags);
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                              unsigned int flags);
hipError_t hipExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags);
hipError_t hipExtStreamCreateWithCUMask(hipStream_t* stream, uint32_t cuMaskSize,
                                        const uint32_t* cuMask);
hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask);
hipError_t hipExternalMemoryGetMappedBuffer(void** devPtr, hipExternalMemory_t extMem,
                                            const hipExternalMemoryBufferDesc* bufferDesc);
hipError_t hipFree(void* ptr);
hipError_t hipFreeArray(hipArray_t array);
hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream);
hipError_t hipFreeHost(void* ptr);
hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);
hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc);
hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func);
hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value);
hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config);
hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config);
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList);
hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array);
hipError_t hipGetDevice(int* deviceId);
hipError_t hipGetDeviceCount(int* count);
hipError_t hipGetDeviceFlags(unsigned int* flags);
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600* prop, int deviceId);
hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop, int device);
hipError_t hipGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags,
                                  hipDriverEntryPointQueryResult* status);
hipError_t hipGetDriverEntryPoint_spt(const char* symbol, void** funcPtr, unsigned long long flags,
                                      hipDriverEntryPointQueryResult* status);
const char* hipGetErrorName(hipError_t hip_error);
const char* hipGetErrorString(hipError_t hipError);
hipError_t hipGetLastError(void);
hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray,
                                     hipMipmappedArray_const_t mipmappedArray, unsigned int level);
hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol);
hipError_t hipGetSymbolSize(size_t* size, const void* symbol);
hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* texref);
hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject);
hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc,
                                               hipTextureObject_t textureObject);
hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
                                          hipTextureObject_t textureObject);
hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipGraph_t childGraph);
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies);
hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                const hipGraphNode_t* pDependencies, size_t numDependencies);
hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                      const hipGraphNode_t* pDependencies, size_t numDependencies,
                                      hipEvent_t event);
hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    hipEvent_t event);
hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                               const hipGraphNode_t* pDependencies, size_t numDependencies,
                               const hipHostNodeParams* pNodeParams);
hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams);
hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   hipMemAllocNodeParams* pNodeParams);
hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  void* dev_ptr);
hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams);
hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                   const hipGraphNode_t* pDependencies, size_t numDependencies,
                                   void* dst, const void* src, size_t count, hipMemcpyKind kind);
hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                           const hipGraphNode_t* pDependencies,
                                           size_t numDependencies, void* dst, const void* symbol,
                                           size_t count, size_t offset, hipMemcpyKind kind);
hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                         const hipGraphNode_t* pDependencies,
                                         size_t numDependencies, const void* symbol,
                                         const void* src, size_t count, size_t offset,
                                         hipMemcpyKind kind);
hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams);
hipError_t hipGraphAddNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                           const hipGraphNode_t* pDependencies, size_t numDependencies,
                           hipGraphNodeParams* nodeParams);
hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph);
hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph);
hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags);
hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags);
hipError_t hipGraphDestroy(hipGraph_t graph);
hipError_t hipGraphDestroyNode(hipGraphNode_t node);
hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);
hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);
hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                               hipGraph_t childGraph);
hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);
hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               hipEvent_t event);
hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                             hipEvent_t event);
hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                         const hipHostNodeParams* pNodeParams);
hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipKernelNodeParams* pNodeParams);
hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           hipMemcpy3DParms* pNodeParams);
hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                             void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind);
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind);
hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                   const void* symbol, const void* src,
                                                   size_t count, size_t offset, hipMemcpyKind kind);
hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                           const hipMemsetParams* pNodeParams);
hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                              hipGraphNode_t* hErrorNode_out,
                              hipGraphExecUpdateResult* updateResult_out);
hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to,
                            size_t* numEdges);
hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes);
hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes,
                                size_t* pNumRootNodes);
hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams);
hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams);
hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                        unsigned long long flags);
hipError_t hipGraphInstantiateWithParams(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                         hipGraphInstantiateParams* instantiateParams);
hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst);
hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          hipKernelNodeAttrValue* value);
hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams);
hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr,
                                          const hipKernelNodeAttrValue* value);
hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams* pNodeParams);
hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);
hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams);
hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr);
hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams);
hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams);
hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src,
                                         size_t count, hipMemcpyKind kind);
hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol,
                                                 size_t count, size_t offset, hipMemcpyKind kind);
hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol,
                                               const void* src, size_t count, size_t offset,
                                               hipMemcpyKind kind);
hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams);
hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams);
hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode,
                                   hipGraph_t clonedGraph);
hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                       size_t* pNumDependencies);
hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                         size_t* pNumDependentNodes);
hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int* isEnabled);
hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType);
hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                  unsigned int isEnabled);
hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count);
hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                      const hipGraphNode_t* to, size_t numDependencies);
hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count,
                                    unsigned int flags);
hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream);
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags);
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target,
                                      unsigned int flags);
hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream);
hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size,
                                               hipGraphicsResource_t resource);
hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource,
                                                unsigned int arrayIndex, unsigned int mipLevel);
hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources,
                                     hipStream_t stream);
hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource);
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void* ptr);
hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
hipError_t hipHostUnregister(void* hostPtr);
hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out,
                                   const hipExternalMemoryHandleDesc* memHandleDesc);
hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,
                                      const hipExternalSemaphoreHandleDesc* semHandleDesc);
hipError_t hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const hipMemsetParams* memsetParams, hipCtx_t ctx);
hipError_t hipInit(unsigned int flags);
hipError_t hipIpcCloseMemHandle(void* devPtr);
hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);
const char* hipKernelNameRef(const hipFunction_t f);
const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream);
hipError_t hipLaunchByPtr(const void* func);
hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, unsigned int sharedMemBytes,
                                      hipStream_t stream);
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                                 unsigned int flags);
hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData);
hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                           void** args, size_t sharedMemBytes, hipStream_t stream);
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent);
hipError_t hipMalloc3DArray(hipArray_t* array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags);
hipError_t hipMallocArray(hipArray_t* array, const hipChannelFormatDesc* desc, size_t width,
                          size_t height, unsigned int flags);
hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream);
hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool,
                                  hipStream_t stream);
hipError_t hipMallocHost(void** ptr, size_t size);
hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags);
hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray,
                                   const struct hipChannelFormatDesc* desc, struct hipExtent extent,
                                   unsigned int numLevels, unsigned int flags);
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
hipError_t hipMemAddressFree(void* devPtr, size_t size);
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags);
hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device);
hipError_t hipMemAdvise_v2(const void* dev_ptr, size_t count, hipMemoryAdvise advice,
                           hipMemLocation location);
hipError_t hipMemAllocHost(void** ptr, size_t size);
hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height,
                            unsigned int elementSizeBytes);
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size,
                        const hipMemAllocationProp* prop, unsigned long long flags);
hipError_t hipMemExportToShareableHandle(void* shareableHandle,
                                         hipMemGenericAllocationHandle_t handle,
                                         hipMemAllocationHandleType handleType,
                                         unsigned long long flags);
hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr);
hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);
hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop,
                                          hipMemAllocationGranularity_flags option);
hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                   hipMemGenericAllocationHandle_t handle);
hipError_t hipMemGetInfo(size_t* free, size_t* total);
hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle,
                                           hipMemAllocationHandleType shHandleType);
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
                     unsigned long long flags);
hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int count,
                               hipStream_t stream);
hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props);
hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool);
hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr);
hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool,
                                             hipMemAllocationHandleType handle_type,
                                             unsigned int flags);
hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool,
                               hipMemLocation* location);
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle,
                                               hipMemAllocationHandleType handle_type,
                                               unsigned int flags);
hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t mem_pool,
                                   hipMemPoolPtrExportData* export_data);
hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list,
                               size_t count);
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold);
hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream);
hipError_t hipMemPrefetchAsync_v2(const void* dev_ptr, size_t count, hipMemLocation location,
                                  unsigned int flags, hipStream_t stream);
hipError_t hipMemPtrGetInfo(void* ptr, size_t* size);
hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count);
hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes,
                                    hipMemRangeAttribute* attributes, size_t num_attributes,
                                    const void* dev_ptr, size_t count);
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr);
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);
hipError_t hipMemUnmap(void* ptr, size_t size);
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpy2DFromArray(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                size_t hOffset, size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DFromArrayAsync(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                     size_t hOffset, size_t width, size_t height,
                                     hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DToArrayAsync(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                   size_t spitch, size_t width, size_t height, hipMemcpyKind kind,
                                   hipStream_t stream);
hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p);
hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                          hipStream_t stream);
hipError_t hipMemcpyAtoH(void* dst, hipArray_t srcArray, size_t srcOffset, size_t count);
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes,
                              hipStream_t stream);
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset,
                              size_t count, hipMemcpyKind kind);
hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                               hipMemcpyKind kind);
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                    hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyHtoA(hipArray_t dstArray, size_t dstOffset, const void* srcHost, size_t count);
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, const void* src, size_t sizeBytes);
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, const void* src, size_t sizeBytes,
                              hipStream_t stream);
hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy);
hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream);
hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId,
                         size_t sizeBytes);
hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice,
                              size_t sizeBytes, hipStream_t stream);
hipError_t hipMemcpyToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                            size_t count, hipMemcpyKind kind);
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset,
                             hipMemcpyKind kind);
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes,
                                  size_t offset, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hipStream_t stream);
hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height);
hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,
                            hipStream_t stream);
hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent);
hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                            hipStream_t stream);
hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count);
hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count,
                             hipStream_t stream);
hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count);
hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream);
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count);
hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count,
                            hipStream_t stream);
hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle,
                                   HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                   unsigned int numMipmapLevels);
hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);
hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray, hipMipmappedArray_t hMipMappedArray,
                                     unsigned int level);
hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
hipError_t hipModuleGetFunctionCount(unsigned int* count, hipModule_t mod);
hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                              const char* name);
hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name);
hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX,
                                            unsigned int gridDimY, unsigned int gridDimZ,
                                            unsigned int blockDimX, unsigned int blockDimY,
                                            unsigned int blockDimZ, unsigned int sharedMemBytes,
                                            hipStream_t stream, void** kernelParams);
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices, unsigned int flags);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra);
hipError_t hipModuleLoadFatBinary(hipModule_t* module, const void* fatbin);
hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions,
                               hipJitOption* options, void** optionValues);
hipError_t hipLinkAddData(hipLinkState_t state, hipJitInputType type, void* data, size_t size,
                          const char* name, unsigned int numOptions, hipJitOption* options,
                          void** optionValues);
hipError_t hipLinkAddFile(hipLinkState_t state, hipJitInputType type, const char* path,
                          unsigned int numOptions, hipJitOption* options, void** optionValues);
hipError_t hipLinkComplete(hipLinkState_t state, void** hipBinOut, size_t* sizeOut);
hipError_t hipLinkCreate(unsigned int numOptions, hipJitOption* options, void** optionValues,
                         hipLinkState_t* stateOut);
hipError_t hipLinkDestroy(hipLinkState_t state);
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f,
                                                              int blockSize,
                                                              size_t dynSharedMemPerBlk);
hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags);
hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, hipFunction_t f,
                                                   size_t dynSharedMemPerBlk, int blockSizeLimit);
hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
                                                            hipFunction_t f,
                                                            size_t dynSharedMemPerBlk,
                                                            int blockSizeLimit, unsigned int flags);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                        int blockSize, size_t dynSharedMemPerBlk);
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* f,
                                                                 int blockSize,
                                                                 size_t dynSharedMemPerBlk,
                                                                 unsigned int flags);
hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, const void* f,
                                             size_t dynSharedMemPerBlk, int blockSizeLimit);
hipError_t hipPeekAtLastError(void);
hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
hipError_t hipPointerSetAttribute(const void* value, hipPointer_attribute attribute,
                                  hipDeviceptr_t ptr);
hipError_t hipProfilerStart();
hipError_t hipProfilerStop();
hipError_t hipRuntimeGetVersion(int* runtimeVersion);
hipError_t hipSetDevice(int deviceId);
hipError_t hipSetDeviceFlags(unsigned flags);
hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                            const hipExternalSemaphoreSignalParams* paramsArray,
                                            unsigned int numExtSems, hipStream_t stream);
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData,
                                unsigned int flags);
hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length,
                                   unsigned int flags);
hipError_t hipStreamCopyAttributes(hipStream_t dst, hipStream_t src);
hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);
hipError_t hipStreamCreate(hipStream_t* stream);
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);
hipError_t hipStreamDestroy(hipStream_t stream);
hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph);
hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                   unsigned long long* pId);
hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out,
                                      unsigned long long* id_out, hipGraph_t* graph_out,
                                      const hipGraphNode_t** dependencies_out,
                                      size_t* numDependencies_out);
hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device);
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);
hipError_t hipStreamGetId(hipStream_t stream, unsigned long long* streamId);
hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);
hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);
hipError_t hipStreamQuery(hipStream_t stream);
hipError_t hipStreamSynchronize(hipStream_t stream);
hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies,
                                              size_t numDependencies, unsigned int flags);
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags,
                                uint32_t mask);
hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
                                uint64_t mask);
hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags);
hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags);
hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams* paramArray, unsigned int flags);
hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject, const HIP_RESOURCE_DESC* pResDesc,
                              const HIP_TEXTURE_DESC* pTexDesc,
                              const HIP_RESOURCE_VIEW_DESC* pResViewDesc);
hipError_t hipTexObjectDestroy(hipTextureObject_t texObject);
hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc, hipTextureObject_t texObject);
hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc,
                                           hipTextureObject_t texObject);
hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc, hipTextureObject_t texObject);
hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr, const textureReference* texRef);
hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode* pam, const textureReference* texRef,
                                   int dim);
hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode* pfm, const textureReference* texRef);
hipError_t hipTexRefGetFlags(unsigned int* pFlags, const textureReference* texRef);
hipError_t hipTexRefGetFormat(hipArray_Format* pFormat, int* pNumChannels,
                              const textureReference* texRef);
hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio, const textureReference* texRef);
hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray, const textureReference* texRef);
hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode* pfm,
                                        const textureReference* texRef);
hipError_t hipTexRefGetMipmapLevelBias(float* pbias, const textureReference* texRef);
hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp,
                                        const textureReference* texRef);
hipError_t hipTexRefSetAddress(size_t* ByteOffset, textureReference* texRef, hipDeviceptr_t dptr,
                               size_t bytes);
hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc,
                                 hipDeviceptr_t dptr, size_t Pitch);
hipError_t hipTexRefSetAddressMode(textureReference* texRef, int dim,
                                   enum hipTextureAddressMode am);
hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags);
hipError_t hipTexRefSetBorderColor(textureReference* texRef, float* pBorderColor);
hipError_t hipTexRefSetFilterMode(textureReference* texRef, enum hipTextureFilterMode fm);
hipError_t hipTexRefSetFlags(textureReference* texRef, unsigned int Flags);
hipError_t hipTexRefSetFormat(textureReference* texRef, hipArray_Format fmt,
                              int NumPackedComponents);
hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef, unsigned int maxAniso);
hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef, enum hipTextureFilterMode fm);
hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef, float bias);
hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef, float minMipMapLevelClamp,
                                        float maxMipMapLevelClamp);
hipError_t hipTexRefSetMipmappedArray(textureReference* texRef,
                                      struct hipMipmappedArray* mipmappedArray, unsigned int Flags);
hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode);
hipError_t hipUnbindTexture(const textureReference* tex);
hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy,
                               unsigned int initialRefcount, unsigned int flags);
hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count);
hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count);
hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray,
                                          const hipExternalSemaphoreWaitParams* paramsArray,
                                          unsigned int numExtSems, hipStream_t stream);
hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f);
hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject, const hipResourceDesc* pResDesc);
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flag);
hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent);
hipError_t hipMemcpy_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
hipError_t hipMemcpyToSymbol_spt(const void* symbol, const void* src, size_t sizeBytes,
                                 size_t offset, hipMemcpyKind kind);
hipError_t hipMemcpyFromSymbol_spt(void* dst, const void* symbol, size_t sizeBytes, size_t offset,
                                   hipMemcpyKind kind);
hipError_t hipMemcpy2D_spt(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                           size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DFromArray_spt(void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset,
                                    size_t hOffset, size_t width, size_t height,
                                    hipMemcpyKind kind);
hipError_t hipMemcpy3D_spt(const struct hipMemcpy3DParms* p);
hipError_t hipMemset_spt(void* dst, int value, size_t sizeBytes);
hipError_t hipMemsetAsync_spt(void* dst, int value, size_t sizeBytes, hipStream_t stream);
hipError_t hipMemset2D_spt(void* dst, size_t pitch, int value, size_t width, size_t height);
hipError_t hipMemset2DAsync_spt(void* dst, size_t pitch, int value, size_t width, size_t height,
                                hipStream_t stream);
hipError_t hipMemset3DAsync_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                                hipStream_t stream);
hipError_t hipMemset3D_spt(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent);
hipError_t hipMemcpyAsync_spt(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                              hipStream_t stream);
hipError_t hipMemcpy3DAsync_spt(const hipMemcpy3DParms* p, hipStream_t stream);
hipError_t hipMemcpy2DAsync_spt(void* dst, size_t dpitch, const void* src, size_t spitch,
                                size_t width, size_t height, hipMemcpyKind kind,
                                hipStream_t stream);
hipError_t hipMemcpyFromSymbolAsync_spt(void* dst, const void* symbol, size_t sizeBytes,
                                        size_t offset, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyToSymbolAsync_spt(const void* symbol, const void* src, size_t sizeBytes,
                                      size_t offset, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpyFromArray_spt(void* dst, hipArray_const_t src, size_t wOffsetSrc,
                                  size_t hOffset, size_t count, hipMemcpyKind kind);
hipError_t hipMemcpy2DToArray_spt(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src,
                                  size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipMemcpy2DFromArrayAsync_spt(void* dst, size_t dpitch, hipArray_const_t src,
                                         size_t wOffsetSrc, size_t hOffsetSrc, size_t width,
                                         size_t height, hipMemcpyKind kind, hipStream_t stream);
hipError_t hipMemcpy2DToArrayAsync_spt(hipArray_t dst, size_t wOffset, size_t hOffset,
                                       const void* src, size_t spitch, size_t width, size_t height,
                                       hipMemcpyKind kind, hipStream_t stream);
hipError_t hipStreamQuery_spt(hipStream_t stream);
hipError_t hipStreamSynchronize_spt(hipStream_t stream);
hipError_t hipStreamGetPriority_spt(hipStream_t stream, int* priority);
hipError_t hipStreamWaitEvent_spt(hipStream_t stream, hipEvent_t event, unsigned int flags);
hipError_t hipStreamGetFlags_spt(hipStream_t stream, unsigned int* flags);
hipError_t hipStreamAddCallback_spt(hipStream_t stream, hipStreamCallback_t callback,
                                    void* userData, unsigned int flags);
hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream = NULL);
hipError_t hipLaunchCooperativeKernel_spt(const void* f, dim3 gridDim, dim3 blockDim,
                                          void** kernelParams, uint32_t sharedMemBytes,
                                          hipStream_t hStream);

hipError_t hipLaunchKernel_spt(const void* function_address, dim3 numBlocks, dim3 dimBlocks,
                               void** args, size_t sharedMemBytes, hipStream_t stream);

hipError_t hipGraphLaunch_spt(hipGraphExec_t graphExec, hipStream_t stream);
hipError_t hipStreamBeginCapture_spt(hipStream_t stream, hipStreamCaptureMode mode);
hipError_t hipStreamEndCapture_spt(hipStream_t stream, hipGraph_t* pGraph);
hipError_t hipStreamIsCapturing_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);
hipError_t hipStreamGetCaptureInfo_spt(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus,
                                       unsigned long long* pId);
hipError_t hipStreamGetCaptureInfo_v2_spt(hipStream_t stream,
                                          hipStreamCaptureStatus* captureStatus_out,
                                          unsigned long long* id_out, hipGraph_t* graph_out,
                                          const hipGraphNode_t** dependencies_out,
                                          size_t* numDependencies_out);
hipError_t hipLaunchHostFunc_spt(hipStream_t stream, hipHostFn_t fn, void* userData);
int hipGetStreamDeviceId(hipStream_t stream);
hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                    const hipGraphNode_t* dependencies, size_t numDependencies,
                                    const HIP_MEMCPY3D* copyParams, hipCtx_t ctx);
hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol);
hipError_t hipGraphAddExternalSemaphoresSignalNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreSignalNodeParams* nodeParams);
hipError_t hipGraphAddExternalSemaphoresWaitNode(
    hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies,
    size_t numDependencies, const hipExternalSemaphoreWaitNodeParams* nodeParams);
hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreSignalNodeParams* nodeParams);
hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
    const hipExternalSemaphoreWaitNodeParams* nodeParams);
hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                                     void* dst, const void* symbol, size_t count,
                                                     size_t offset, hipMemcpyKind kind);
hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out);
hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams);
hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(
    hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out);
hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(
    hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams);
hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                       unsigned int numDevices, unsigned int flags);
hipError_t hipExtGetLastError();
hipError_t hipTexRefGetBorderColor(float* pBorderColor, const textureReference* texRef);
hipError_t hipTexRefGetArray(hipArray_t* pArray, const textureReference* texRef);
hipError_t hipGetProcAddress(const char* symbol, void** pfn, int hipVersion, uint64_t flags,
                             hipDriverProcAddressQueryResult* symbolStatus = NULL);
hipError_t hipStreamBeginCaptureToGraph(hipStream_t stream, hipGraph_t graph,
                                        const hipGraphNode_t* dependencies,
                                        const hipGraphEdgeData* dependencyData,
                                        size_t numDependencies, hipStreamCaptureMode mode);
hipError_t hipGetFuncBySymbol(hipFunction_t* functionPtr, const void* symbolPtr);
hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     hipDeviceptr_t dptr);
hipError_t hipDrvGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const HIP_MEMCPY3D* copyParams, hipCtx_t ctx);
hipError_t hipDrvGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                              const hipMemsetParams* memsetParams, hipCtx_t ctx);
hipError_t hipSetValidDevices(int* device_arr, int len);
hipError_t hipMemcpyAtoD(hipDeviceptr_t dstDevice, hipArray_t srcArray, size_t srcOffset,
                         size_t ByteCount);
hipError_t hipMemcpyDtoA(hipArray_t dstArray, size_t dstOffset, hipDeviceptr_t srcDevice,
                         size_t ByteCount);
hipError_t hipMemcpyAtoA(hipArray_t dstArray, size_t dstOffset, hipArray_t srcArray,
                         size_t srcOffset, size_t ByteCount);
hipError_t hipMemcpyAtoHAsync(void* dstHost, hipArray_t srcArray, size_t srcOffset,
                              size_t ByteCount, hipStream_t stream);
hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset, const void* srcHost,
                              size_t ByteCount, hipStream_t stream);
hipError_t hipMemcpy2DArrayToArray(hipArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                   hipArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc,
                                   size_t width, size_t height, hipMemcpyKind kind);
hipError_t hipGraphExecGetFlags(hipGraphExec_t graphExec, unsigned long long* flags);
hipError_t hipGraphNodeSetParams(hipGraphNode_t node, hipGraphNodeParams* nodeParams);
hipError_t hipGraphExecNodeSetParams(hipGraphExec_t graphExec, hipGraphNode_t node,
                                     hipGraphNodeParams* nodeParams);
hipError_t hipExternalMemoryGetMappedMipmappedArray(
    hipMipmappedArray_t* mipmap, hipExternalMemory_t extMem,
    const hipExternalMemoryMipmappedArrayDesc* mipmapDesc);
hipError_t hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode, HIP_MEMCPY3D* nodeParams);
hipError_t hipDrvGraphMemcpyNodeSetParams(hipGraphNode_t hNode, const HIP_MEMCPY3D* nodeParams);
hipError_t hipGraphAddBatchMemOpNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph,
                                     const hipGraphNode_t* dependencies, size_t numDependencies,
                                     const hipBatchMemOpNodeParams* nodeParams);
hipError_t hipGraphBatchMemOpNodeGetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams_out);
hipError_t hipGraphBatchMemOpNodeSetParams(hipGraphNode_t hNode,
                                           hipBatchMemOpNodeParams* nodeParams);
hipError_t hipGraphExecBatchMemOpNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode,
                                               const hipBatchMemOpNodeParams* nodeParams);
hipError_t hipEventRecordWithFlags(hipEvent_t event, hipStream_t stream, unsigned flags);
hipError_t hipLaunchKernelExC(const hipLaunchConfig_t* config, const void* fPtr, void** args);
hipError_t hipDrvLaunchKernelEx(const HIP_LAUNCH_CONFIG* config, hipFunction_t f, void** params,
                                void** extra);
hipError_t hipMemGetHandleForAddressRange(void* handle, hipDeviceptr_t dptr, size_t size,
                                          hipMemRangeHandleType handleType,
                                          unsigned long long flags);
hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch, unsigned char value, size_t width,
                         size_t height);
hipError_t hipMemsetD2D8Async(hipDeviceptr_t dst, size_t dstPitch, unsigned char value,
                              size_t width, size_t height, hipStream_t stream);
hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch, unsigned short value, size_t width,
                          size_t height);
hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch, unsigned short value,
                               size_t width, size_t height, hipStream_t stream);
hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch, unsigned int value, size_t width,
                          size_t height);
hipError_t hipMemsetD2D32Async(hipDeviceptr_t dst, size_t dstPitch, unsigned int value,
                               size_t width, size_t height, hipStream_t stream);
hipError_t hipStreamGetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 hipStreamAttrValue* value);
hipError_t hipStreamSetAttribute(hipStream_t stream, hipStreamAttrID attr,
                                 const hipStreamAttrValue* value);
hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count,
                               hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                               size_t* failIdx, hipStream_t stream);
hipError_t hipMemcpy3DBatchAsync(size_t numOps, struct hipMemcpy3DBatchOp* opList, size_t* failIdx,
                                 unsigned long long flags, hipStream_t stream);
hipError_t hipMemcpy3DPeer(hipMemcpy3DPeerParms* p);
hipError_t hipMemcpy3DPeerAsync(hipMemcpy3DPeerParms* p, hipStream_t stream);
hipError_t hipLibraryLoadData(hipLibrary_t* library, const void* code, hipJitOption* jitOptions,
                              void** jitOptionsValues, unsigned int numJitOptions,
                              hipLibraryOption* libraryOptions, void** libraryOptionValues,
                              unsigned int numLibraryOptions);
hipError_t hipLibraryLoadFromFile(hipLibrary_t* library, const char* fileName,
                                  hipJitOption* jitOptions, void** jitOptionsValues,
                                  unsigned int numJitOptions, hipLibraryOption* libraryOptions,
                                  void** libraryOptionValues, unsigned int numLibraryOptions);
hipError_t hipLibraryUnload(hipLibrary_t library);
hipError_t hipLibraryGetKernel(hipKernel_t* pKernel, hipLibrary_t library, const char* name);
hipError_t hipLibraryGetKernelCount(unsigned int* count, hipLibrary_t library);
}  // namespace hip

namespace hip {
namespace {
void UpdateDispatchTable(HipCompilerDispatchTable* ptrCompilerDispatchTable) {
  ptrCompilerDispatchTable->size = sizeof(HipCompilerDispatchTable);
  ptrCompilerDispatchTable->__hipPopCallConfiguration_fn = hip::__hipPopCallConfiguration;
  ptrCompilerDispatchTable->__hipPushCallConfiguration_fn = hip::__hipPushCallConfiguration;
  ptrCompilerDispatchTable->__hipRegisterFatBinary_fn = hip::__hipRegisterFatBinary;
  ptrCompilerDispatchTable->__hipRegisterFunction_fn = hip::__hipRegisterFunction;
  ptrCompilerDispatchTable->__hipRegisterManagedVar_fn = hip::__hipRegisterManagedVar;
  ptrCompilerDispatchTable->__hipRegisterSurface_fn = hip::__hipRegisterSurface;
  ptrCompilerDispatchTable->__hipRegisterTexture_fn = hip::__hipRegisterTexture;
  ptrCompilerDispatchTable->__hipRegisterVar_fn = hip::__hipRegisterVar;
  ptrCompilerDispatchTable->__hipUnregisterFatBinary_fn = hip::__hipUnregisterFatBinary;
}

void UpdateDispatchTable(HipToolsDispatchTable* ptrToolsDispatchTable) {
  ptrToolsDispatchTable->size = sizeof(HipToolsDispatchTable);
  ptrToolsDispatchTable->__hipReportDevices_fn = nullptr;
}

void UpdateDispatchTable(HipDispatchTable* ptrDispatchTable) {
  ptrDispatchTable->size = sizeof(HipDispatchTable);
  ptrDispatchTable->hipApiName_fn = hip::hipApiName;
  ptrDispatchTable->hipArray3DCreate_fn = hip::hipArray3DCreate;
  ptrDispatchTable->hipArray3DGetDescriptor_fn = hip::hipArray3DGetDescriptor;
  ptrDispatchTable->hipArrayCreate_fn = hip::hipArrayCreate;
  ptrDispatchTable->hipArrayDestroy_fn = hip::hipArrayDestroy;
  ptrDispatchTable->hipArrayGetDescriptor_fn = hip::hipArrayGetDescriptor;
  ptrDispatchTable->hipArrayGetInfo_fn = hip::hipArrayGetInfo;
  ptrDispatchTable->hipBindTexture_fn = hip::hipBindTexture;
  ptrDispatchTable->hipBindTexture2D_fn = hip::hipBindTexture2D;
  ptrDispatchTable->hipBindTextureToArray_fn = hip::hipBindTextureToArray;
  ptrDispatchTable->hipBindTextureToMipmappedArray_fn = hip::hipBindTextureToMipmappedArray;
  ptrDispatchTable->hipChooseDevice_fn = hip::hipChooseDevice;
  ptrDispatchTable->hipChooseDeviceR0000_fn = hip::hipChooseDeviceR0000;
  ptrDispatchTable->hipConfigureCall_fn = hip::hipConfigureCall;
  ptrDispatchTable->hipCreateSurfaceObject_fn = hip::hipCreateSurfaceObject;
  ptrDispatchTable->hipCreateTextureObject_fn = hip::hipCreateTextureObject;
  ptrDispatchTable->hipCtxCreate_fn = hip::hipCtxCreate;
  ptrDispatchTable->hipCtxDestroy_fn = hip::hipCtxDestroy;
  ptrDispatchTable->hipCtxDisablePeerAccess_fn = hip::hipCtxDisablePeerAccess;
  ptrDispatchTable->hipCtxEnablePeerAccess_fn = hip::hipCtxEnablePeerAccess;
  ptrDispatchTable->hipCtxGetApiVersion_fn = hip::hipCtxGetApiVersion;
  ptrDispatchTable->hipCtxGetCacheConfig_fn = hip::hipCtxGetCacheConfig;
  ptrDispatchTable->hipCtxGetCurrent_fn = hip::hipCtxGetCurrent;
  ptrDispatchTable->hipCtxGetDevice_fn = hip::hipCtxGetDevice;
  ptrDispatchTable->hipCtxGetFlags_fn = hip::hipCtxGetFlags;
  ptrDispatchTable->hipCtxGetSharedMemConfig_fn = hip::hipCtxGetSharedMemConfig;
  ptrDispatchTable->hipCtxPopCurrent_fn = hip::hipCtxPopCurrent;
  ptrDispatchTable->hipCtxPushCurrent_fn = hip::hipCtxPushCurrent;
  ptrDispatchTable->hipCtxSetCacheConfig_fn = hip::hipCtxSetCacheConfig;
  ptrDispatchTable->hipCtxSetCurrent_fn = hip::hipCtxSetCurrent;
  ptrDispatchTable->hipCtxSetSharedMemConfig_fn = hip::hipCtxSetSharedMemConfig;
  ptrDispatchTable->hipCtxSynchronize_fn = hip::hipCtxSynchronize;
  ptrDispatchTable->hipDestroyExternalMemory_fn = hip::hipDestroyExternalMemory;
  ptrDispatchTable->hipDestroyExternalSemaphore_fn = hip::hipDestroyExternalSemaphore;
  ptrDispatchTable->hipDestroySurfaceObject_fn = hip::hipDestroySurfaceObject;
  ptrDispatchTable->hipDestroyTextureObject_fn = hip::hipDestroyTextureObject;
  ptrDispatchTable->hipDeviceCanAccessPeer_fn = hip::hipDeviceCanAccessPeer;
  ptrDispatchTable->hipDeviceComputeCapability_fn = hip::hipDeviceComputeCapability;
  ptrDispatchTable->hipDeviceDisablePeerAccess_fn = hip::hipDeviceDisablePeerAccess;
  ptrDispatchTable->hipDeviceEnablePeerAccess_fn = hip::hipDeviceEnablePeerAccess;
  ptrDispatchTable->hipDeviceGet_fn = hip::hipDeviceGet;
  ptrDispatchTable->hipDeviceGetAttribute_fn = hip::hipDeviceGetAttribute;
  ptrDispatchTable->hipDeviceGetByPCIBusId_fn = hip::hipDeviceGetByPCIBusId;
  ptrDispatchTable->hipDeviceGetCacheConfig_fn = hip::hipDeviceGetCacheConfig;
  ptrDispatchTable->hipDeviceGetDefaultMemPool_fn = hip::hipDeviceGetDefaultMemPool;
  ptrDispatchTable->hipDeviceGetGraphMemAttribute_fn = hip::hipDeviceGetGraphMemAttribute;
  ptrDispatchTable->hipDeviceGetLimit_fn = hip::hipDeviceGetLimit;
  ptrDispatchTable->hipDeviceGetMemPool_fn = hip::hipDeviceGetMemPool;
  ptrDispatchTable->hipDeviceGetName_fn = hip::hipDeviceGetName;
  ptrDispatchTable->hipDeviceGetP2PAttribute_fn = hip::hipDeviceGetP2PAttribute;
  ptrDispatchTable->hipDeviceGetPCIBusId_fn = hip::hipDeviceGetPCIBusId;
  ptrDispatchTable->hipDeviceGetSharedMemConfig_fn = hip::hipDeviceGetSharedMemConfig;
  ptrDispatchTable->hipDeviceGetStreamPriorityRange_fn = hip::hipDeviceGetStreamPriorityRange;
  ptrDispatchTable->hipDeviceGetTexture1DLinearMaxWidth_fn =
      hip::hipDeviceGetTexture1DLinearMaxWidth;
  ptrDispatchTable->hipDeviceGetUuid_fn = hip::hipDeviceGetUuid;
  ptrDispatchTable->hipDeviceGraphMemTrim_fn = hip::hipDeviceGraphMemTrim;
  ptrDispatchTable->hipDevicePrimaryCtxGetState_fn = hip::hipDevicePrimaryCtxGetState;
  ptrDispatchTable->hipDevicePrimaryCtxRelease_fn = hip::hipDevicePrimaryCtxRelease;
  ptrDispatchTable->hipDevicePrimaryCtxReset_fn = hip::hipDevicePrimaryCtxReset;
  ptrDispatchTable->hipDevicePrimaryCtxRetain_fn = hip::hipDevicePrimaryCtxRetain;
  ptrDispatchTable->hipDevicePrimaryCtxSetFlags_fn = hip::hipDevicePrimaryCtxSetFlags;
  ptrDispatchTable->hipDeviceReset_fn = hip::hipDeviceReset;
  ptrDispatchTable->hipDeviceSetCacheConfig_fn = hip::hipDeviceSetCacheConfig;
  ptrDispatchTable->hipDeviceSetGraphMemAttribute_fn = hip::hipDeviceSetGraphMemAttribute;
  ptrDispatchTable->hipDeviceSetLimit_fn = hip::hipDeviceSetLimit;
  ptrDispatchTable->hipDeviceSetMemPool_fn = hip::hipDeviceSetMemPool;
  ptrDispatchTable->hipDeviceSetSharedMemConfig_fn = hip::hipDeviceSetSharedMemConfig;
  ptrDispatchTable->hipDeviceSynchronize_fn = hip::hipDeviceSynchronize;
  ptrDispatchTable->hipDeviceTotalMem_fn = hip::hipDeviceTotalMem;
  ptrDispatchTable->hipDriverGetVersion_fn = hip::hipDriverGetVersion;
  ptrDispatchTable->hipDrvGetErrorName_fn = hip::hipDrvGetErrorName;
  ptrDispatchTable->hipDrvGetErrorString_fn = hip::hipDrvGetErrorString;
  ptrDispatchTable->hipDrvGraphAddMemcpyNode_fn = hip::hipDrvGraphAddMemcpyNode;
  ptrDispatchTable->hipDrvMemcpy2DUnaligned_fn = hip::hipDrvMemcpy2DUnaligned;
  ptrDispatchTable->hipDrvMemcpy3D_fn = hip::hipDrvMemcpy3D;
  ptrDispatchTable->hipDrvMemcpy3DAsync_fn = hip::hipDrvMemcpy3DAsync;
  ptrDispatchTable->hipDrvPointerGetAttributes_fn = hip::hipDrvPointerGetAttributes;
  ptrDispatchTable->hipEventCreate_fn = hip::hipEventCreate;
  ptrDispatchTable->hipEventCreateWithFlags_fn = hip::hipEventCreateWithFlags;
  ptrDispatchTable->hipEventDestroy_fn = hip::hipEventDestroy;
  ptrDispatchTable->hipEventElapsedTime_fn = hip::hipEventElapsedTime;
  ptrDispatchTable->hipEventQuery_fn = hip::hipEventQuery;
  ptrDispatchTable->hipEventRecord_fn = hip::hipEventRecord;
  ptrDispatchTable->hipEventSynchronize_fn = hip::hipEventSynchronize;
  ptrDispatchTable->hipExtGetLinkTypeAndHopCount_fn = hip::hipExtGetLinkTypeAndHopCount;
  ptrDispatchTable->hipExtLaunchKernel_fn = hip::hipExtLaunchKernel;
  ptrDispatchTable->hipExtLaunchMultiKernelMultiDevice_fn = hip::hipExtLaunchMultiKernelMultiDevice;
  ptrDispatchTable->hipExtMallocWithFlags_fn = hip::hipExtMallocWithFlags;
  ptrDispatchTable->hipExtStreamCreateWithCUMask_fn = hip::hipExtStreamCreateWithCUMask;
  ptrDispatchTable->hipExtStreamGetCUMask_fn = hip::hipExtStreamGetCUMask;
  ptrDispatchTable->hipExternalMemoryGetMappedBuffer_fn = hip::hipExternalMemoryGetMappedBuffer;
  ptrDispatchTable->hipFree_fn = hip::hipFree;
  ptrDispatchTable->hipFreeArray_fn = hip::hipFreeArray;
  ptrDispatchTable->hipFreeAsync_fn = hip::hipFreeAsync;
  ptrDispatchTable->hipFreeHost_fn = hip::hipFreeHost;
  ptrDispatchTable->hipFreeMipmappedArray_fn = hip::hipFreeMipmappedArray;
  ptrDispatchTable->hipFuncGetAttribute_fn = hip::hipFuncGetAttribute;
  ptrDispatchTable->hipFuncGetAttributes_fn = hip::hipFuncGetAttributes;
  ptrDispatchTable->hipFuncSetAttribute_fn = hip::hipFuncSetAttribute;
  ptrDispatchTable->hipFuncSetCacheConfig_fn = hip::hipFuncSetCacheConfig;
  ptrDispatchTable->hipFuncSetSharedMemConfig_fn = hip::hipFuncSetSharedMemConfig;
  ptrDispatchTable->hipGLGetDevices_fn = hip::hipGLGetDevices;
  ptrDispatchTable->hipGetChannelDesc_fn = hip::hipGetChannelDesc;
  ptrDispatchTable->hipGetDevice_fn = hip::hipGetDevice;
  ptrDispatchTable->hipGetDeviceCount_fn = hip::hipGetDeviceCount;
  ptrDispatchTable->hipGetDeviceFlags_fn = hip::hipGetDeviceFlags;
  ptrDispatchTable->hipGetDevicePropertiesR0600_fn = hip::hipGetDevicePropertiesR0600;
  ptrDispatchTable->hipGetErrorName_fn = hip::hipGetErrorName;
  ptrDispatchTable->hipGetErrorString_fn = hip::hipGetErrorString;
  ptrDispatchTable->hipGetLastError_fn = hip::hipGetLastError;
  ptrDispatchTable->hipGetMipmappedArrayLevel_fn = hip::hipGetMipmappedArrayLevel;
  ptrDispatchTable->hipGetSymbolAddress_fn = hip::hipGetSymbolAddress;
  ptrDispatchTable->hipGetSymbolSize_fn = hip::hipGetSymbolSize;
  ptrDispatchTable->hipGetTextureAlignmentOffset_fn = hip::hipGetTextureAlignmentOffset;
  ptrDispatchTable->hipGetTextureObjectResourceDesc_fn = hip::hipGetTextureObjectResourceDesc;
  ptrDispatchTable->hipGetTextureObjectResourceViewDesc_fn =
      hip::hipGetTextureObjectResourceViewDesc;
  ptrDispatchTable->hipGetTextureObjectTextureDesc_fn = hip::hipGetTextureObjectTextureDesc;
  ptrDispatchTable->hipGetTextureReference_fn = hip::hipGetTextureReference;
  ptrDispatchTable->hipGraphAddChildGraphNode_fn = hip::hipGraphAddChildGraphNode;
  ptrDispatchTable->hipGraphAddDependencies_fn = hip::hipGraphAddDependencies;
  ptrDispatchTable->hipGraphAddEmptyNode_fn = hip::hipGraphAddEmptyNode;
  ptrDispatchTable->hipGraphAddEventRecordNode_fn = hip::hipGraphAddEventRecordNode;
  ptrDispatchTable->hipGraphAddEventWaitNode_fn = hip::hipGraphAddEventWaitNode;
  ptrDispatchTable->hipGraphAddHostNode_fn = hip::hipGraphAddHostNode;
  ptrDispatchTable->hipGraphAddKernelNode_fn = hip::hipGraphAddKernelNode;
  ptrDispatchTable->hipGraphAddMemAllocNode_fn = hip::hipGraphAddMemAllocNode;
  ptrDispatchTable->hipGraphAddMemFreeNode_fn = hip::hipGraphAddMemFreeNode;
  ptrDispatchTable->hipGraphAddMemcpyNode_fn = hip::hipGraphAddMemcpyNode;
  ptrDispatchTable->hipGraphAddMemcpyNode1D_fn = hip::hipGraphAddMemcpyNode1D;
  ptrDispatchTable->hipGraphAddMemcpyNodeFromSymbol_fn = hip::hipGraphAddMemcpyNodeFromSymbol;
  ptrDispatchTable->hipGraphAddMemcpyNodeToSymbol_fn = hip::hipGraphAddMemcpyNodeToSymbol;
  ptrDispatchTable->hipGraphAddMemsetNode_fn = hip::hipGraphAddMemsetNode;
  ptrDispatchTable->hipGraphAddNode_fn = hip::hipGraphAddNode;
  ptrDispatchTable->hipGraphChildGraphNodeGetGraph_fn = hip::hipGraphChildGraphNodeGetGraph;
  ptrDispatchTable->hipGraphClone_fn = hip::hipGraphClone;
  ptrDispatchTable->hipGraphCreate_fn = hip::hipGraphCreate;
  ptrDispatchTable->hipGraphDebugDotPrint_fn = hip::hipGraphDebugDotPrint;
  ptrDispatchTable->hipGraphDestroy_fn = hip::hipGraphDestroy;
  ptrDispatchTable->hipGraphDestroyNode_fn = hip::hipGraphDestroyNode;
  ptrDispatchTable->hipGraphEventRecordNodeGetEvent_fn = hip::hipGraphEventRecordNodeGetEvent;
  ptrDispatchTable->hipGraphEventRecordNodeSetEvent_fn = hip::hipGraphEventRecordNodeSetEvent;
  ptrDispatchTable->hipGraphEventWaitNodeGetEvent_fn = hip::hipGraphEventWaitNodeGetEvent;
  ptrDispatchTable->hipGraphEventWaitNodeSetEvent_fn = hip::hipGraphEventWaitNodeSetEvent;
  ptrDispatchTable->hipGraphExecChildGraphNodeSetParams_fn =
      hip::hipGraphExecChildGraphNodeSetParams;
  ptrDispatchTable->hipGraphExecDestroy_fn = hip::hipGraphExecDestroy;
  ptrDispatchTable->hipGraphExecEventRecordNodeSetEvent_fn =
      hip::hipGraphExecEventRecordNodeSetEvent;
  ptrDispatchTable->hipGraphExecEventWaitNodeSetEvent_fn = hip::hipGraphExecEventWaitNodeSetEvent;
  ptrDispatchTable->hipGraphExecHostNodeSetParams_fn = hip::hipGraphExecHostNodeSetParams;
  ptrDispatchTable->hipGraphExecKernelNodeSetParams_fn = hip::hipGraphExecKernelNodeSetParams;
  ptrDispatchTable->hipGraphExecMemcpyNodeSetParams_fn = hip::hipGraphExecMemcpyNodeSetParams;
  ptrDispatchTable->hipGraphExecMemcpyNodeSetParams1D_fn = hip::hipGraphExecMemcpyNodeSetParams1D;
  ptrDispatchTable->hipGraphExecMemcpyNodeSetParamsFromSymbol_fn =
      hip::hipGraphExecMemcpyNodeSetParamsFromSymbol;
  ptrDispatchTable->hipGraphExecMemcpyNodeSetParamsToSymbol_fn =
      hip::hipGraphExecMemcpyNodeSetParamsToSymbol;
  ptrDispatchTable->hipGraphExecMemsetNodeSetParams_fn = hip::hipGraphExecMemsetNodeSetParams;
  ptrDispatchTable->hipGraphExecUpdate_fn = hip::hipGraphExecUpdate;
  ptrDispatchTable->hipGraphGetEdges_fn = hip::hipGraphGetEdges;
  ptrDispatchTable->hipGraphGetNodes_fn = hip::hipGraphGetNodes;
  ptrDispatchTable->hipGraphGetRootNodes_fn = hip::hipGraphGetRootNodes;
  ptrDispatchTable->hipGraphHostNodeGetParams_fn = hip::hipGraphHostNodeGetParams;
  ptrDispatchTable->hipGraphHostNodeSetParams_fn = hip::hipGraphHostNodeSetParams;
  ptrDispatchTable->hipGraphInstantiate_fn = hip::hipGraphInstantiate;
  ptrDispatchTable->hipGraphInstantiateWithFlags_fn = hip::hipGraphInstantiateWithFlags;
  ptrDispatchTable->hipGraphInstantiateWithParams_fn = hip::hipGraphInstantiateWithParams;
  ptrDispatchTable->hipGraphKernelNodeCopyAttributes_fn = hip::hipGraphKernelNodeCopyAttributes;
  ptrDispatchTable->hipGraphKernelNodeGetAttribute_fn = hip::hipGraphKernelNodeGetAttribute;
  ptrDispatchTable->hipGraphKernelNodeGetParams_fn = hip::hipGraphKernelNodeGetParams;
  ptrDispatchTable->hipGraphKernelNodeSetAttribute_fn = hip::hipGraphKernelNodeSetAttribute;
  ptrDispatchTable->hipGraphKernelNodeSetParams_fn = hip::hipGraphKernelNodeSetParams;
  ptrDispatchTable->hipGraphLaunch_fn = hip::hipGraphLaunch;
  ptrDispatchTable->hipGraphMemAllocNodeGetParams_fn = hip::hipGraphMemAllocNodeGetParams;
  ptrDispatchTable->hipGraphMemFreeNodeGetParams_fn = hip::hipGraphMemFreeNodeGetParams;
  ptrDispatchTable->hipGraphMemcpyNodeGetParams_fn = hip::hipGraphMemcpyNodeGetParams;
  ptrDispatchTable->hipGraphMemcpyNodeSetParams_fn = hip::hipGraphMemcpyNodeSetParams;
  ptrDispatchTable->hipGraphMemcpyNodeSetParams1D_fn = hip::hipGraphMemcpyNodeSetParams1D;
  ptrDispatchTable->hipGraphMemcpyNodeSetParamsFromSymbol_fn =
      hip::hipGraphMemcpyNodeSetParamsFromSymbol;
  ptrDispatchTable->hipGraphMemcpyNodeSetParamsToSymbol_fn =
      hip::hipGraphMemcpyNodeSetParamsToSymbol;
  ptrDispatchTable->hipGraphMemsetNodeGetParams_fn = hip::hipGraphMemsetNodeGetParams;
  ptrDispatchTable->hipGraphMemsetNodeSetParams_fn = hip::hipGraphMemsetNodeSetParams;
  ptrDispatchTable->hipGraphNodeFindInClone_fn = hip::hipGraphNodeFindInClone;
  ptrDispatchTable->hipGraphNodeGetDependencies_fn = hip::hipGraphNodeGetDependencies;
  ptrDispatchTable->hipGraphNodeGetDependentNodes_fn = hip::hipGraphNodeGetDependentNodes;
  ptrDispatchTable->hipGraphNodeGetEnabled_fn = hip::hipGraphNodeGetEnabled;
  ptrDispatchTable->hipGraphNodeGetType_fn = hip::hipGraphNodeGetType;
  ptrDispatchTable->hipGraphNodeSetEnabled_fn = hip::hipGraphNodeSetEnabled;
  ptrDispatchTable->hipGraphReleaseUserObject_fn = hip::hipGraphReleaseUserObject;
  ptrDispatchTable->hipGraphRemoveDependencies_fn = hip::hipGraphRemoveDependencies;
  ptrDispatchTable->hipGraphRetainUserObject_fn = hip::hipGraphRetainUserObject;
  ptrDispatchTable->hipGraphUpload_fn = hip::hipGraphUpload;
  ptrDispatchTable->hipGraphicsGLRegisterBuffer_fn = hip::hipGraphicsGLRegisterBuffer;
  ptrDispatchTable->hipGraphicsGLRegisterImage_fn = hip::hipGraphicsGLRegisterImage;
  ptrDispatchTable->hipGraphicsMapResources_fn = hip::hipGraphicsMapResources;
  ptrDispatchTable->hipGraphicsResourceGetMappedPointer_fn =
      hip::hipGraphicsResourceGetMappedPointer;
  ptrDispatchTable->hipGraphicsSubResourceGetMappedArray_fn =
      hip::hipGraphicsSubResourceGetMappedArray;
  ptrDispatchTable->hipGraphicsUnmapResources_fn = hip::hipGraphicsUnmapResources;
  ptrDispatchTable->hipGraphicsUnregisterResource_fn = hip::hipGraphicsUnregisterResource;
  ptrDispatchTable->hipHostAlloc_fn = hip::hipHostAlloc;
  ptrDispatchTable->hipHostFree_fn = hip::hipHostFree;
  ptrDispatchTable->hipHostGetDevicePointer_fn = hip::hipHostGetDevicePointer;
  ptrDispatchTable->hipHostGetFlags_fn = hip::hipHostGetFlags;
  ptrDispatchTable->hipHostMalloc_fn = hip::hipHostMalloc;
  ptrDispatchTable->hipExtHostAlloc_fn = nullptr;
  ptrDispatchTable->hipHostRegister_fn = hip::hipHostRegister;
  ptrDispatchTable->hipHostUnregister_fn = hip::hipHostUnregister;
  ptrDispatchTable->hipImportExternalMemory_fn = hip::hipImportExternalMemory;
  ptrDispatchTable->hipImportExternalSemaphore_fn = hip::hipImportExternalSemaphore;
  ptrDispatchTable->hipInit_fn = hip::hipInit;
  ptrDispatchTable->hipIpcCloseMemHandle_fn = hip::hipIpcCloseMemHandle;
  ptrDispatchTable->hipIpcGetEventHandle_fn = hip::hipIpcGetEventHandle;
  ptrDispatchTable->hipIpcGetMemHandle_fn = hip::hipIpcGetMemHandle;
  ptrDispatchTable->hipIpcOpenEventHandle_fn = hip::hipIpcOpenEventHandle;
  ptrDispatchTable->hipIpcOpenMemHandle_fn = hip::hipIpcOpenMemHandle;
  ptrDispatchTable->hipKernelNameRef_fn = hip::hipKernelNameRef;
  ptrDispatchTable->hipKernelNameRefByPtr_fn = hip::hipKernelNameRefByPtr;
  ptrDispatchTable->hipLaunchByPtr_fn = hip::hipLaunchByPtr;
  ptrDispatchTable->hipLaunchCooperativeKernel_fn = hip::hipLaunchCooperativeKernel;
  ptrDispatchTable->hipLaunchCooperativeKernelMultiDevice_fn =
      hip::hipLaunchCooperativeKernelMultiDevice;
  ptrDispatchTable->hipLaunchHostFunc_fn = hip::hipLaunchHostFunc;
  ptrDispatchTable->hipLaunchKernel_fn = hip::hipLaunchKernel;
  ptrDispatchTable->hipMalloc_fn = hip::hipMalloc;
  ptrDispatchTable->hipMalloc3D_fn = hip::hipMalloc3D;
  ptrDispatchTable->hipMalloc3DArray_fn = hip::hipMalloc3DArray;
  ptrDispatchTable->hipMallocArray_fn = hip::hipMallocArray;
  ptrDispatchTable->hipMallocAsync_fn = hip::hipMallocAsync;
  ptrDispatchTable->hipMallocFromPoolAsync_fn = hip::hipMallocFromPoolAsync;
  ptrDispatchTable->hipMallocHost_fn = hip::hipMallocHost;
  ptrDispatchTable->hipMallocManaged_fn = hip::hipMallocManaged;
  ptrDispatchTable->hipMallocMipmappedArray_fn = hip::hipMallocMipmappedArray;
  ptrDispatchTable->hipMallocPitch_fn = hip::hipMallocPitch;
  ptrDispatchTable->hipMemAddressFree_fn = hip::hipMemAddressFree;
  ptrDispatchTable->hipMemAddressReserve_fn = hip::hipMemAddressReserve;
  ptrDispatchTable->hipMemAdvise_fn = hip::hipMemAdvise;
  ptrDispatchTable->hipMemAdvise_v2_fn = hip::hipMemAdvise_v2;
  ptrDispatchTable->hipMemAllocHost_fn = hip::hipMemAllocHost;
  ptrDispatchTable->hipMemAllocPitch_fn = hip::hipMemAllocPitch;
  ptrDispatchTable->hipMemCreate_fn = hip::hipMemCreate;
  ptrDispatchTable->hipMemExportToShareableHandle_fn = hip::hipMemExportToShareableHandle;
  ptrDispatchTable->hipMemGetAccess_fn = hip::hipMemGetAccess;
  ptrDispatchTable->hipMemGetAddressRange_fn = hip::hipMemGetAddressRange;
  ptrDispatchTable->hipMemGetAllocationGranularity_fn = hip::hipMemGetAllocationGranularity;
  ptrDispatchTable->hipMemGetAllocationPropertiesFromHandle_fn =
      hip::hipMemGetAllocationPropertiesFromHandle;
  ptrDispatchTable->hipMemGetInfo_fn = hip::hipMemGetInfo;
  ptrDispatchTable->hipMemImportFromShareableHandle_fn = hip::hipMemImportFromShareableHandle;
  ptrDispatchTable->hipMemMap_fn = hip::hipMemMap;
  ptrDispatchTable->hipMemMapArrayAsync_fn = hip::hipMemMapArrayAsync;
  ptrDispatchTable->hipMemPoolCreate_fn = hip::hipMemPoolCreate;
  ptrDispatchTable->hipMemPoolDestroy_fn = hip::hipMemPoolDestroy;
  ptrDispatchTable->hipMemPoolExportPointer_fn = hip::hipMemPoolExportPointer;
  ptrDispatchTable->hipMemPoolExportToShareableHandle_fn = hip::hipMemPoolExportToShareableHandle;
  ptrDispatchTable->hipMemPoolGetAccess_fn = hip::hipMemPoolGetAccess;
  ptrDispatchTable->hipMemPoolGetAttribute_fn = hip::hipMemPoolGetAttribute;
  ptrDispatchTable->hipMemPoolImportFromShareableHandle_fn =
      hip::hipMemPoolImportFromShareableHandle;
  ptrDispatchTable->hipMemPoolImportPointer_fn = hip::hipMemPoolImportPointer;
  ptrDispatchTable->hipMemPoolSetAccess_fn = hip::hipMemPoolSetAccess;
  ptrDispatchTable->hipMemPoolSetAttribute_fn = hip::hipMemPoolSetAttribute;
  ptrDispatchTable->hipMemPoolTrimTo_fn = hip::hipMemPoolTrimTo;
  ptrDispatchTable->hipMemPrefetchAsync_fn = hip::hipMemPrefetchAsync;
  ptrDispatchTable->hipMemPrefetchAsync_v2_fn = hip::hipMemPrefetchAsync_v2;
  ptrDispatchTable->hipMemPtrGetInfo_fn = hip::hipMemPtrGetInfo;
  ptrDispatchTable->hipMemRangeGetAttribute_fn = hip::hipMemRangeGetAttribute;
  ptrDispatchTable->hipMemRangeGetAttributes_fn = hip::hipMemRangeGetAttributes;
  ptrDispatchTable->hipMemRelease_fn = hip::hipMemRelease;
  ptrDispatchTable->hipMemRetainAllocationHandle_fn = hip::hipMemRetainAllocationHandle;
  ptrDispatchTable->hipMemSetAccess_fn = hip::hipMemSetAccess;
  ptrDispatchTable->hipMemUnmap_fn = hip::hipMemUnmap;
  ptrDispatchTable->hipMemcpy_fn = hip::hipMemcpy;
  ptrDispatchTable->hipMemcpy2D_fn = hip::hipMemcpy2D;
  ptrDispatchTable->hipMemcpy2DAsync_fn = hip::hipMemcpy2DAsync;
  ptrDispatchTable->hipMemcpy2DFromArray_fn = hip::hipMemcpy2DFromArray;
  ptrDispatchTable->hipMemcpy2DFromArrayAsync_fn = hip::hipMemcpy2DFromArrayAsync;
  ptrDispatchTable->hipMemcpy2DToArray_fn = hip::hipMemcpy2DToArray;
  ptrDispatchTable->hipMemcpy2DToArrayAsync_fn = hip::hipMemcpy2DToArrayAsync;
  ptrDispatchTable->hipMemcpy3D_fn = hip::hipMemcpy3D;
  ptrDispatchTable->hipMemcpy3DAsync_fn = hip::hipMemcpy3DAsync;
  ptrDispatchTable->hipMemcpyAsync_fn = hip::hipMemcpyAsync;
  ptrDispatchTable->hipMemcpyAtoH_fn = hip::hipMemcpyAtoH;
  ptrDispatchTable->hipMemcpyDtoD_fn = hip::hipMemcpyDtoD;
  ptrDispatchTable->hipMemcpyDtoDAsync_fn = hip::hipMemcpyDtoDAsync;
  ptrDispatchTable->hipMemcpyDtoH_fn = hip::hipMemcpyDtoH;
  ptrDispatchTable->hipMemcpyDtoHAsync_fn = hip::hipMemcpyDtoHAsync;
  ptrDispatchTable->hipMemcpyFromArray_fn = hip::hipMemcpyFromArray;
  ptrDispatchTable->hipMemcpyFromSymbol_fn = hip::hipMemcpyFromSymbol;
  ptrDispatchTable->hipMemcpyFromSymbolAsync_fn = hip::hipMemcpyFromSymbolAsync;
  ptrDispatchTable->hipMemcpyHtoA_fn = hip::hipMemcpyHtoA;
  ptrDispatchTable->hipMemcpyHtoD_fn = hip::hipMemcpyHtoD;
  ptrDispatchTable->hipMemcpyHtoDAsync_fn = hip::hipMemcpyHtoDAsync;
  ptrDispatchTable->hipMemcpyParam2D_fn = hip::hipMemcpyParam2D;
  ptrDispatchTable->hipMemcpyParam2DAsync_fn = hip::hipMemcpyParam2DAsync;
  ptrDispatchTable->hipMemcpyPeer_fn = hip::hipMemcpyPeer;
  ptrDispatchTable->hipMemcpyPeerAsync_fn = hip::hipMemcpyPeerAsync;
  ptrDispatchTable->hipMemcpyToArray_fn = hip::hipMemcpyToArray;
  ptrDispatchTable->hipMemcpyToSymbol_fn = hip::hipMemcpyToSymbol;
  ptrDispatchTable->hipMemcpyToSymbolAsync_fn = hip::hipMemcpyToSymbolAsync;
  ptrDispatchTable->hipMemcpyWithStream_fn = hip::hipMemcpyWithStream;
  ptrDispatchTable->hipMemset_fn = hip::hipMemset;
  ptrDispatchTable->hipMemset2D_fn = hip::hipMemset2D;
  ptrDispatchTable->hipMemset2DAsync_fn = hip::hipMemset2DAsync;
  ptrDispatchTable->hipMemset3D_fn = hip::hipMemset3D;
  ptrDispatchTable->hipMemset3DAsync_fn = hip::hipMemset3DAsync;
  ptrDispatchTable->hipMemsetAsync_fn = hip::hipMemsetAsync;
  ptrDispatchTable->hipMemsetD16_fn = hip::hipMemsetD16;
  ptrDispatchTable->hipMemsetD16Async_fn = hip::hipMemsetD16Async;
  ptrDispatchTable->hipMemsetD32_fn = hip::hipMemsetD32;
  ptrDispatchTable->hipMemsetD32Async_fn = hip::hipMemsetD32Async;
  ptrDispatchTable->hipMemsetD8_fn = hip::hipMemsetD8;
  ptrDispatchTable->hipMemsetD8Async_fn = hip::hipMemsetD8Async;
  ptrDispatchTable->hipMipmappedArrayCreate_fn = hip::hipMipmappedArrayCreate;
  ptrDispatchTable->hipMipmappedArrayDestroy_fn = hip::hipMipmappedArrayDestroy;
  ptrDispatchTable->hipMipmappedArrayGetLevel_fn = hip::hipMipmappedArrayGetLevel;
  ptrDispatchTable->hipModuleGetFunction_fn = hip::hipModuleGetFunction;
  ptrDispatchTable->hipModuleGetFunctionCount_fn = hip::hipModuleGetFunctionCount;
  ptrDispatchTable->hipModuleGetGlobal_fn = hip::hipModuleGetGlobal;
  ptrDispatchTable->hipModuleGetTexRef_fn = hip::hipModuleGetTexRef;
  ptrDispatchTable->hipModuleLaunchCooperativeKernel_fn = hip::hipModuleLaunchCooperativeKernel;
  ptrDispatchTable->hipModuleLaunchCooperativeKernelMultiDevice_fn =
      hip::hipModuleLaunchCooperativeKernelMultiDevice;
  ptrDispatchTable->hipModuleLaunchKernel_fn = hip::hipModuleLaunchKernel;
  ptrDispatchTable->hipModuleLoadFatBinary_fn = hip::hipModuleLoadFatBinary;
  ptrDispatchTable->hipModuleLoad_fn = hip::hipModuleLoad;
  ptrDispatchTable->hipModuleLoadData_fn = hip::hipModuleLoadData;
  ptrDispatchTable->hipModuleLoadDataEx_fn = hip::hipModuleLoadDataEx;
  ptrDispatchTable->hipLinkAddData_fn = hip::hipLinkAddData;
  ptrDispatchTable->hipLinkAddFile_fn = hip::hipLinkAddFile;
  ptrDispatchTable->hipLinkComplete_fn = hip::hipLinkComplete;
  ptrDispatchTable->hipLinkCreate_fn = hip::hipLinkCreate;
  ptrDispatchTable->hipLinkDestroy_fn = hip::hipLinkDestroy;
  ptrDispatchTable->hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn =
      hip::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
  ptrDispatchTable->hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn =
      hip::hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
  ptrDispatchTable->hipModuleOccupancyMaxPotentialBlockSize_fn =
      hip::hipModuleOccupancyMaxPotentialBlockSize;
  ptrDispatchTable->hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn =
      hip::hipModuleOccupancyMaxPotentialBlockSizeWithFlags;
  ptrDispatchTable->hipModuleUnload_fn = hip::hipModuleUnload;
  ptrDispatchTable->hipOccupancyMaxActiveBlocksPerMultiprocessor_fn =
      hip::hipOccupancyMaxActiveBlocksPerMultiprocessor;
  ptrDispatchTable->hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn =
      hip::hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
  ptrDispatchTable->hipOccupancyMaxPotentialBlockSize_fn = hip::hipOccupancyMaxPotentialBlockSize;
  ptrDispatchTable->hipPeekAtLastError_fn = hip::hipPeekAtLastError;
  ptrDispatchTable->hipPointerGetAttribute_fn = hip::hipPointerGetAttribute;
  ptrDispatchTable->hipPointerGetAttributes_fn = hip::hipPointerGetAttributes;
  ptrDispatchTable->hipPointerSetAttribute_fn = hip::hipPointerSetAttribute;
  ptrDispatchTable->hipProfilerStart_fn = hip::hipProfilerStart;
  ptrDispatchTable->hipProfilerStop_fn = hip::hipProfilerStop;
  ptrDispatchTable->hipRuntimeGetVersion_fn = hip::hipRuntimeGetVersion;
  ptrDispatchTable->hipSetDevice_fn = hip::hipSetDevice;
  ptrDispatchTable->hipSetDeviceFlags_fn = hip::hipSetDeviceFlags;
  ptrDispatchTable->hipSetupArgument_fn = hip::hipSetupArgument;
  ptrDispatchTable->hipSignalExternalSemaphoresAsync_fn = hip::hipSignalExternalSemaphoresAsync;
  ptrDispatchTable->hipStreamAddCallback_fn = hip::hipStreamAddCallback;
  ptrDispatchTable->hipStreamAttachMemAsync_fn = hip::hipStreamAttachMemAsync;
  ptrDispatchTable->hipStreamBeginCapture_fn = hip::hipStreamBeginCapture;
  ptrDispatchTable->hipStreamCopyAttributes_fn = hip::hipStreamCopyAttributes;
  ptrDispatchTable->hipStreamCreate_fn = hip::hipStreamCreate;
  ptrDispatchTable->hipStreamCreateWithFlags_fn = hip::hipStreamCreateWithFlags;
  ptrDispatchTable->hipStreamCreateWithPriority_fn = hip::hipStreamCreateWithPriority;
  ptrDispatchTable->hipStreamDestroy_fn = hip::hipStreamDestroy;
  ptrDispatchTable->hipStreamEndCapture_fn = hip::hipStreamEndCapture;
  ptrDispatchTable->hipStreamGetCaptureInfo_fn = hip::hipStreamGetCaptureInfo;
  ptrDispatchTable->hipStreamGetCaptureInfo_v2_fn = hip::hipStreamGetCaptureInfo_v2;
  ptrDispatchTable->hipStreamGetDevice_fn = hip::hipStreamGetDevice;
  ptrDispatchTable->hipStreamGetFlags_fn = hip::hipStreamGetFlags;
  ptrDispatchTable->hipStreamGetId_fn = hip::hipStreamGetId;
  ptrDispatchTable->hipStreamGetPriority_fn = hip::hipStreamGetPriority;
  ptrDispatchTable->hipStreamIsCapturing_fn = hip::hipStreamIsCapturing;
  ptrDispatchTable->hipStreamQuery_fn = hip::hipStreamQuery;
  ptrDispatchTable->hipStreamSynchronize_fn = hip::hipStreamSynchronize;
  ptrDispatchTable->hipStreamUpdateCaptureDependencies_fn = hip::hipStreamUpdateCaptureDependencies;
  ptrDispatchTable->hipStreamWaitEvent_fn = hip::hipStreamWaitEvent;
  ptrDispatchTable->hipStreamWaitValue32_fn = hip::hipStreamWaitValue32;
  ptrDispatchTable->hipStreamWaitValue64_fn = hip::hipStreamWaitValue64;
  ptrDispatchTable->hipStreamWriteValue32_fn = hip::hipStreamWriteValue32;
  ptrDispatchTable->hipStreamWriteValue64_fn = hip::hipStreamWriteValue64;
  ptrDispatchTable->hipStreamBatchMemOp_fn = hip::hipStreamBatchMemOp;
  ptrDispatchTable->hipTexObjectCreate_fn = hip::hipTexObjectCreate;
  ptrDispatchTable->hipTexObjectDestroy_fn = hip::hipTexObjectDestroy;
  ptrDispatchTable->hipTexObjectGetResourceDesc_fn = hip::hipTexObjectGetResourceDesc;
  ptrDispatchTable->hipTexObjectGetResourceViewDesc_fn = hip::hipTexObjectGetResourceViewDesc;
  ptrDispatchTable->hipTexObjectGetTextureDesc_fn = hip::hipTexObjectGetTextureDesc;
  ptrDispatchTable->hipTexRefGetAddress_fn = hip::hipTexRefGetAddress;
  ptrDispatchTable->hipTexRefGetAddressMode_fn = hip::hipTexRefGetAddressMode;
  ptrDispatchTable->hipTexRefGetFilterMode_fn = hip::hipTexRefGetFilterMode;
  ptrDispatchTable->hipTexRefGetFlags_fn = hip::hipTexRefGetFlags;
  ptrDispatchTable->hipTexRefGetFormat_fn = hip::hipTexRefGetFormat;
  ptrDispatchTable->hipTexRefGetMaxAnisotropy_fn = hip::hipTexRefGetMaxAnisotropy;
  ptrDispatchTable->hipTexRefGetMipMappedArray_fn = hip::hipTexRefGetMipMappedArray;
  ptrDispatchTable->hipTexRefGetMipmapFilterMode_fn = hip::hipTexRefGetMipmapFilterMode;
  ptrDispatchTable->hipTexRefGetMipmapLevelBias_fn = hip::hipTexRefGetMipmapLevelBias;
  ptrDispatchTable->hipTexRefGetMipmapLevelClamp_fn = hip::hipTexRefGetMipmapLevelClamp;
  ptrDispatchTable->hipTexRefSetAddress_fn = hip::hipTexRefSetAddress;
  ptrDispatchTable->hipTexRefSetAddress2D_fn = hip::hipTexRefSetAddress2D;
  ptrDispatchTable->hipTexRefSetAddressMode_fn = hip::hipTexRefSetAddressMode;
  ptrDispatchTable->hipTexRefSetArray_fn = hip::hipTexRefSetArray;
  ptrDispatchTable->hipTexRefSetBorderColor_fn = hip::hipTexRefSetBorderColor;
  ptrDispatchTable->hipTexRefSetFilterMode_fn = hip::hipTexRefSetFilterMode;
  ptrDispatchTable->hipTexRefSetFlags_fn = hip::hipTexRefSetFlags;
  ptrDispatchTable->hipTexRefSetFormat_fn = hip::hipTexRefSetFormat;
  ptrDispatchTable->hipTexRefSetMaxAnisotropy_fn = hip::hipTexRefSetMaxAnisotropy;
  ptrDispatchTable->hipTexRefSetMipmapFilterMode_fn = hip::hipTexRefSetMipmapFilterMode;
  ptrDispatchTable->hipTexRefSetMipmapLevelBias_fn = hip::hipTexRefSetMipmapLevelBias;
  ptrDispatchTable->hipTexRefSetMipmapLevelClamp_fn = hip::hipTexRefSetMipmapLevelClamp;
  ptrDispatchTable->hipTexRefSetMipmappedArray_fn = hip::hipTexRefSetMipmappedArray;
  ptrDispatchTable->hipThreadExchangeStreamCaptureMode_fn = hip::hipThreadExchangeStreamCaptureMode;
  ptrDispatchTable->hipUnbindTexture_fn = hip::hipUnbindTexture;
  ptrDispatchTable->hipUserObjectCreate_fn = hip::hipUserObjectCreate;
  ptrDispatchTable->hipUserObjectRelease_fn = hip::hipUserObjectRelease;
  ptrDispatchTable->hipUserObjectRetain_fn = hip::hipUserObjectRetain;
  ptrDispatchTable->hipWaitExternalSemaphoresAsync_fn = hip::hipWaitExternalSemaphoresAsync;
  ptrDispatchTable->hipCreateChannelDesc_fn = hip::hipCreateChannelDesc;
  ptrDispatchTable->hipCreateSurfaceObject_fn = hip::hipCreateSurfaceObject;
  ptrDispatchTable->hipExtModuleLaunchKernel_fn = hip::hipExtModuleLaunchKernel;
  ptrDispatchTable->hipHccModuleLaunchKernel_fn = hip::hipHccModuleLaunchKernel;
  ptrDispatchTable->hipMemcpy_spt_fn = hip::hipMemcpy_spt;
  ptrDispatchTable->hipMemcpyToSymbol_spt_fn = hip::hipMemcpyToSymbol_spt;
  ptrDispatchTable->hipMemcpyFromSymbol_spt_fn = hip::hipMemcpyFromSymbol_spt;
  ptrDispatchTable->hipMemcpy2D_spt_fn = hip::hipMemcpy2D_spt;
  ptrDispatchTable->hipMemcpy2DFromArray_spt_fn = hip::hipMemcpy2DFromArray_spt;
  ptrDispatchTable->hipMemcpy3D_spt_fn = hip::hipMemcpy3D_spt;
  ptrDispatchTable->hipMemset_spt_fn = hip::hipMemset_spt;
  ptrDispatchTable->hipMemsetAsync_spt_fn = hip::hipMemsetAsync_spt;
  ptrDispatchTable->hipMemset2D_spt_fn = hip::hipMemset2D_spt;
  ptrDispatchTable->hipMemset2DAsync_spt_fn = hip::hipMemset2DAsync_spt;
  ptrDispatchTable->hipMemset3DAsync_spt_fn = hip::hipMemset3DAsync_spt;
  ptrDispatchTable->hipMemset3D_spt_fn = hip::hipMemset3D_spt;
  ptrDispatchTable->hipMemcpyAsync_spt_fn = hip::hipMemcpyAsync_spt;
  ptrDispatchTable->hipMemcpy3DAsync_spt_fn = hip::hipMemcpy3DAsync_spt;
  ptrDispatchTable->hipMemcpy2DAsync_spt_fn = hip::hipMemcpy2DAsync_spt;
  ptrDispatchTable->hipMemcpyFromSymbolAsync_spt_fn = hip::hipMemcpyFromSymbolAsync_spt;
  ptrDispatchTable->hipMemcpyToSymbolAsync_spt_fn = hip::hipMemcpyToSymbolAsync_spt;
  ptrDispatchTable->hipMemcpyFromArray_spt_fn = hip::hipMemcpyFromArray_spt;
  ptrDispatchTable->hipMemcpy2DToArray_spt_fn = hip::hipMemcpy2DToArray_spt;
  ptrDispatchTable->hipMemcpy2DFromArrayAsync_spt_fn = hip::hipMemcpy2DFromArrayAsync_spt;
  ptrDispatchTable->hipMemcpy2DToArrayAsync_spt_fn = hip::hipMemcpy2DToArrayAsync_spt;
  ptrDispatchTable->hipStreamQuery_spt_fn = hip::hipStreamQuery_spt;
  ptrDispatchTable->hipStreamSynchronize_spt_fn = hip::hipStreamSynchronize_spt;
  ptrDispatchTable->hipStreamGetPriority_spt_fn = hip::hipStreamGetPriority_spt;
  ptrDispatchTable->hipStreamWaitEvent_spt_fn = hip::hipStreamWaitEvent_spt;
  ptrDispatchTable->hipStreamGetFlags_spt_fn = hip::hipStreamGetFlags_spt;
  ptrDispatchTable->hipStreamAddCallback_spt_fn = hip::hipStreamAddCallback_spt;
  ptrDispatchTable->hipEventRecord_spt_fn = hip::hipEventRecord_spt;
  ptrDispatchTable->hipLaunchCooperativeKernel_spt_fn = hip::hipLaunchCooperativeKernel_spt;
  ptrDispatchTable->hipLaunchKernel_spt_fn = hip::hipLaunchKernel_spt;
  ptrDispatchTable->hipGraphLaunch_spt_fn = hip::hipGraphLaunch_spt;
  ptrDispatchTable->hipStreamBeginCapture_spt_fn = hip::hipStreamBeginCapture_spt;
  ptrDispatchTable->hipStreamEndCapture_spt_fn = hip::hipStreamEndCapture_spt;
  ptrDispatchTable->hipStreamIsCapturing_spt_fn = hip::hipStreamIsCapturing_spt;
  ptrDispatchTable->hipStreamGetCaptureInfo_spt_fn = hip::hipStreamGetCaptureInfo_spt;
  ptrDispatchTable->hipStreamGetCaptureInfo_v2_spt_fn = hip::hipStreamGetCaptureInfo_v2_spt;
  ptrDispatchTable->hipLaunchHostFunc_spt_fn = hip::hipLaunchHostFunc_spt;
  ptrDispatchTable->hipGetStreamDeviceId_fn = hip::hipGetStreamDeviceId;
  ptrDispatchTable->hipDrvGraphAddMemsetNode_fn = hip::hipDrvGraphAddMemsetNode;
  ptrDispatchTable->hipGetDevicePropertiesR0000_fn = hip::hipGetDevicePropertiesR0000;
  ptrDispatchTable->hipGetDriverEntryPoint_fn = hip::hipGetDriverEntryPoint;
  ptrDispatchTable->hipGetDriverEntryPoint_spt_fn = hip::hipGetDriverEntryPoint_spt;
  ptrDispatchTable->hipExtGetLastError_fn = hip::hipExtGetLastError;
  ptrDispatchTable->hipTexRefGetBorderColor_fn = hip::hipTexRefGetBorderColor;
  ptrDispatchTable->hipTexRefGetArray_fn = hip::hipTexRefGetArray;
  ptrDispatchTable->hipGetProcAddress_fn = hip::hipGetProcAddress;
  ptrDispatchTable->hipStreamBeginCaptureToGraph_fn = hip::hipStreamBeginCaptureToGraph;
  ptrDispatchTable->hipGetFuncBySymbol_fn = hip::hipGetFuncBySymbol;
  ptrDispatchTable->hipSetValidDevices_fn = hip::hipSetValidDevices;
  ptrDispatchTable->hipMemcpyAtoD_fn = hip::hipMemcpyAtoD;
  ptrDispatchTable->hipMemcpyDtoA_fn = hip::hipMemcpyDtoA;
  ptrDispatchTable->hipMemcpyAtoA_fn = hip::hipMemcpyAtoA;
  ptrDispatchTable->hipMemcpyAtoHAsync_fn = hip::hipMemcpyAtoHAsync;
  ptrDispatchTable->hipMemcpyHtoAAsync_fn = hip::hipMemcpyHtoAAsync;
  ptrDispatchTable->hipMemcpy2DArrayToArray_fn = hip::hipMemcpy2DArrayToArray;
  ptrDispatchTable->hipDrvGraphAddMemFreeNode_fn = hip::hipDrvGraphAddMemFreeNode;
  ptrDispatchTable->hipDrvGraphExecMemcpyNodeSetParams_fn = hip::hipDrvGraphExecMemcpyNodeSetParams;
  ptrDispatchTable->hipDrvGraphExecMemsetNodeSetParams_fn = hip::hipDrvGraphExecMemsetNodeSetParams;
  ptrDispatchTable->hipGraphExecGetFlags_fn = hip::hipGraphExecGetFlags;
  ptrDispatchTable->hipGraphNodeSetParams_fn = hip::hipGraphNodeSetParams;
  ptrDispatchTable->hipGraphExecNodeSetParams_fn = hip::hipGraphExecNodeSetParams;
  ptrDispatchTable->hipExternalMemoryGetMappedMipmappedArray_fn =
      hip::hipExternalMemoryGetMappedMipmappedArray;
  ptrDispatchTable->hipDrvGraphMemcpyNodeGetParams_fn = hip::hipDrvGraphMemcpyNodeGetParams;
  ptrDispatchTable->hipDrvGraphMemcpyNodeSetParams_fn = hip::hipDrvGraphMemcpyNodeSetParams;
  ptrDispatchTable->hipGraphAddBatchMemOpNode_fn = hip::hipGraphAddBatchMemOpNode;
  ptrDispatchTable->hipGraphBatchMemOpNodeGetParams_fn = hip::hipGraphBatchMemOpNodeGetParams;
  ptrDispatchTable->hipGraphBatchMemOpNodeSetParams_fn = hip::hipGraphBatchMemOpNodeSetParams;
  ptrDispatchTable->hipGraphExecBatchMemOpNodeSetParams_fn =
      hip::hipGraphExecBatchMemOpNodeSetParams;
  ptrDispatchTable->hipEventRecordWithFlags_fn = hip::hipEventRecordWithFlags;
  ptrDispatchTable->hipLaunchKernelExC_fn = hip::hipLaunchKernelExC;
  ptrDispatchTable->hipDrvLaunchKernelEx_fn = hip::hipDrvLaunchKernelEx;
  ptrDispatchTable->hipMemGetHandleForAddressRange_fn = hip::hipMemGetHandleForAddressRange;
  ptrDispatchTable->hipMemsetD2D8_fn = hip::hipMemsetD2D8;
  ptrDispatchTable->hipMemsetD2D8Async_fn = hip::hipMemsetD2D8Async;
  ptrDispatchTable->hipMemsetD2D16_fn = hip::hipMemsetD2D16;
  ptrDispatchTable->hipMemsetD2D16Async_fn = hip::hipMemsetD2D16Async;
  ptrDispatchTable->hipMemsetD2D32_fn = hip::hipMemsetD2D32;
  ptrDispatchTable->hipMemsetD2D32Async_fn = hip::hipMemsetD2D32Async;
  ptrDispatchTable->hipStreamGetAttribute_fn = hip::hipStreamGetAttribute;
  ptrDispatchTable->hipStreamSetAttribute_fn = hip::hipStreamSetAttribute;
  ptrDispatchTable->hipMemcpyBatchAsync_fn = hip::hipMemcpyBatchAsync;
  ptrDispatchTable->hipMemcpy3DBatchAsync_fn = hip::hipMemcpy3DBatchAsync;
  ptrDispatchTable->hipMemcpy3DPeer_fn = hip::hipMemcpy3DPeer;
  ptrDispatchTable->hipMemcpy3DPeerAsync_fn = hip::hipMemcpy3DPeerAsync;
  ptrDispatchTable->hipLibraryLoadData_fn = hip::hipLibraryLoadData;
  ptrDispatchTable->hipLibraryLoadFromFile_fn = hip::hipLibraryLoadFromFile;
  ptrDispatchTable->hipLibraryUnload_fn = hip::hipLibraryUnload;
  ptrDispatchTable->hipLibraryGetKernel_fn = hip::hipLibraryGetKernel;
  ptrDispatchTable->hipLibraryGetKernelCount_fn = hip::hipLibraryGetKernelCount;
}

#if HIP_ROCPROFILER_REGISTER > 0
template <typename Tp> struct dispatch_table_info;

#define HIP_DEFINE_DISPATCH_TABLE_INFO(TYPE, NAME)                                                 \
  template <> struct dispatch_table_info<TYPE> {                                                   \
    static constexpr auto name = #NAME;                                                            \
    static constexpr auto version = HIP_ROCP_REG_VERSION;                                          \
    static constexpr auto import_func = &ROCPROFILER_REGISTER_IMPORT_FUNC(NAME);                   \
  };

constexpr auto ComputeTableSize(size_t num_funcs) {
  return (num_funcs * sizeof(void*)) + sizeof(uint64_t);
}

HIP_DEFINE_DISPATCH_TABLE_INFO(HipDispatchTable, hip)
HIP_DEFINE_DISPATCH_TABLE_INFO(HipCompilerDispatchTable, hip_compiler)
HIP_DEFINE_DISPATCH_TABLE_INFO(HipToolsDispatchTable, hip_tools)
#endif

template <typename Tp> void ToolsInit(Tp* table) {
#if HIP_ROCPROFILER_REGISTER > 0
  auto table_array = std::array<void*, 1>{static_cast<void*>(table)};
  auto lib_id = rocprofiler_register_library_indentifier_t{};
  auto rocp_reg_status = rocprofiler_register_library_api_table(
      dispatch_table_info<Tp>::name, dispatch_table_info<Tp>::import_func,
      dispatch_table_info<Tp>::version, table_array.data(), table_array.size(), &lib_id);

  // TODO(jrmadsen): check environment variable? Does this need to compile on Windows?
  bool report_register_errors = false;
  if (report_register_errors && rocp_reg_status != ROCP_REG_SUCCESS)
    fprintf(stderr, "rocprofiler-register failed for %s with error code %i: %s\n",
            dispatch_table_info<Tp>::name, rocp_reg_status,
            rocprofiler_register_error_string(rocp_reg_status));
#else
  (void)table;
#endif
}

template <typename Tp> Tp& GetDispatchTableImpl() {
  // using a static inside a function prevents static initialization fiascos
  static auto dispatch_table = Tp{};

  // Change all the function pointers to point to the HIP runtime implementation functions
  UpdateDispatchTable(&dispatch_table);

  // Profiler Registration, may wrap the function pointers
  ToolsInit(&dispatch_table);

  return dispatch_table;
}
}  // namespace

// At the -O3 optimization level, these functions are vectorized (gcc 11.4.1),
// resulting in extra code to preload the dispatch table onto the stack using vector registers.
// This preloading occurs before verifying if the table has been initialized.
// Since the table is typically already initialized, this adds unnecessary overhead to the critical
// path.
#if defined(__GNUC__) || defined(__clang__)
#define NO_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
#else
#define NO_VECTORIZE
#endif
NO_VECTORIZE const HipDispatchTable* GetHipDispatchTable() {
  static auto* _v = &GetDispatchTableImpl<HipDispatchTable>();
  return _v;
}
NO_VECTORIZE const HipCompilerDispatchTable* GetHipCompilerDispatchTable() {
  static auto* _v = &GetDispatchTableImpl<HipCompilerDispatchTable>();
  return _v;
}
const HipToolsDispatchTable* GetHipToolsDispatchTable() {
  static auto* _v = &GetDispatchTableImpl<HipToolsDispatchTable>();
  return _v;
}
}  // namespace hip

#if !defined(_WIN32)
constexpr auto ComputeTableOffset(size_t num_funcs) {
  return (num_funcs * sizeof(void*)) + sizeof(size_t);
}

// HIP_ENFORCE_ABI_VERSIONING will cause a compiler error if the size of the API table changed (most
// likely due to addition of new dispatch table entry) to make sure the developer is reminded to
// update the table versioning value before changing the value in HIP_ENFORCE_ABI_VERSIONING to make
// this static assert pass.
//
// HIP_ENFORCE_ABI will cause a compiler error if the order of the members in the API table change.
// Do not reorder member variables and change existing HIP_ENFORCE_ABI values -- always
//
// Please note: rocprofiler will do very strict compile time checks to make
// sure these versioning values are appropriately updated -- so commenting out this check, only
// updating the size field in HIP_ENFORCE_ABI_VERSIONING, etc. will result in the
// rocprofiler-sdk failing to build and you will be forced to do the work anyway.
#define HIP_ENFORCE_ABI_VERSIONING(TABLE, NUM)                                                     \
  static_assert(                                                                                   \
      sizeof(TABLE) == ComputeTableOffset(NUM),                                                    \
      "size of the API table struct has changed. Update the STEP_VERSION number (or in rare "      \
      "cases, the MAJOR_VERSION number) in <hipamd/include/hip/amd_detail/hip_api_trace.hpp> for " \
      "the failing API struct before changing the SIZE field passed to "                           \
      "HIP_DEFINE_DISPATCH_TABLE_INFO");

#define HIP_ENFORCE_ABI(TABLE, ENTRY, NUM)                                                         \
  static_assert(offsetof(TABLE, ENTRY) == ComputeTableOffset(NUM),                                 \
                "ABI break for " #TABLE "." #ENTRY                                                 \
                ". Only add new function pointers to end of struct and do not rearrange them ");

// These ensure that function pointers are not re-ordered
// HIP_COMPILER_API_TABLE_STEP_VERSION == 0
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipPopCallConfiguration_fn, 0)
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipPushCallConfiguration_fn, 1)
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipRegisterFatBinary_fn, 2)
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipRegisterFunction_fn, 3)
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipRegisterManagedVar_fn, 4)
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipRegisterSurface_fn, 5)
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipRegisterTexture_fn, 6)
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipRegisterVar_fn, 7)
HIP_ENFORCE_ABI(HipCompilerDispatchTable, __hipUnregisterFatBinary_fn, 8)
// HIP_COMPILER_API_TABLE_STEP_VERSION == 1

// if HIP_ENFORCE_ABI entries are added for each new function pointer in the table, the number below
// will be +1 of the number in the last HIP_ENFORCE_ABI line. E.g.:
//
//  HIP_ENFORCE_ABI(<table>, <functor>, 8)
//
//  HIP_ENFORCE_ABI_VERSIONING(<table>, 9) <- 8 + 1 = 9
HIP_ENFORCE_ABI_VERSIONING(HipCompilerDispatchTable, 9)

static_assert(HIP_COMPILER_API_TABLE_MAJOR_VERSION == 0 && HIP_COMPILER_API_TABLE_STEP_VERSION == 0,
              "If you get this error, add new HIP_ENFORCE_ABI(...) code for the new function "
              "pointers and then update this check so it is true");

// These ensure that function pointers are not re-ordered
// HIP_TOOLS_API_TABLE_STEP_VERSION == 0
HIP_ENFORCE_ABI(HipToolsDispatchTable, __hipReportDevices_fn, 0)
// HIP_TOOLS_API_TABLE_STEP_VERSION == 1

// if HIP_ENFORCE_ABI entries are added for each new function pointer in the table, the number below
// will be +1 of the number in the last HIP_ENFORCE_ABI line. E.g.:
//
//  HIP_ENFORCE_ABI(<table>, <functor>, 8)
//
//  HIP_ENFORCE_ABI_VERSIONING(<table>, 9) <- 8 + 1 = 9
HIP_ENFORCE_ABI_VERSIONING(HipToolsDispatchTable, 1)

static_assert(HIP_TOOLS_API_TABLE_MAJOR_VERSION == 0 && HIP_TOOLS_API_TABLE_STEP_VERSION == 0,
              "If you get this error, add new HIP_ENFORCE_ABI(...) code for the new function "
              "pointers and then update this check so it is true");

// These ensure that function pointers are not re-ordered
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 0
HIP_ENFORCE_ABI(HipDispatchTable, hipApiName_fn, 0)
HIP_ENFORCE_ABI(HipDispatchTable, hipArray3DCreate_fn, 1)
HIP_ENFORCE_ABI(HipDispatchTable, hipArray3DGetDescriptor_fn, 2)
HIP_ENFORCE_ABI(HipDispatchTable, hipArrayCreate_fn, 3)
HIP_ENFORCE_ABI(HipDispatchTable, hipArrayDestroy_fn, 4)
HIP_ENFORCE_ABI(HipDispatchTable, hipArrayGetDescriptor_fn, 5)
HIP_ENFORCE_ABI(HipDispatchTable, hipArrayGetInfo_fn, 6)
HIP_ENFORCE_ABI(HipDispatchTable, hipBindTexture_fn, 7)
HIP_ENFORCE_ABI(HipDispatchTable, hipBindTexture2D_fn, 8)
HIP_ENFORCE_ABI(HipDispatchTable, hipBindTextureToArray_fn, 9)
HIP_ENFORCE_ABI(HipDispatchTable, hipBindTextureToMipmappedArray_fn, 10)
HIP_ENFORCE_ABI(HipDispatchTable, hipChooseDevice_fn, 11)
HIP_ENFORCE_ABI(HipDispatchTable, hipChooseDeviceR0000_fn, 12)
HIP_ENFORCE_ABI(HipDispatchTable, hipConfigureCall_fn, 13)
HIP_ENFORCE_ABI(HipDispatchTable, hipCreateSurfaceObject_fn, 14)
HIP_ENFORCE_ABI(HipDispatchTable, hipCreateTextureObject_fn, 15)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxCreate_fn, 16)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxDestroy_fn, 17)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxDisablePeerAccess_fn, 18)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxEnablePeerAccess_fn, 19)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxGetApiVersion_fn, 20)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxGetCacheConfig_fn, 21)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxGetCurrent_fn, 22)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxGetDevice_fn, 23)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxGetFlags_fn, 24)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxGetSharedMemConfig_fn, 25)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxPopCurrent_fn, 26)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxPushCurrent_fn, 27)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxSetCacheConfig_fn, 28)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxSetCurrent_fn, 29)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxSetSharedMemConfig_fn, 30)
HIP_ENFORCE_ABI(HipDispatchTable, hipCtxSynchronize_fn, 31)
HIP_ENFORCE_ABI(HipDispatchTable, hipDestroyExternalMemory_fn, 32)
HIP_ENFORCE_ABI(HipDispatchTable, hipDestroyExternalSemaphore_fn, 33)
HIP_ENFORCE_ABI(HipDispatchTable, hipDestroySurfaceObject_fn, 34)
HIP_ENFORCE_ABI(HipDispatchTable, hipDestroyTextureObject_fn, 35)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceCanAccessPeer_fn, 36)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceComputeCapability_fn, 37)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceDisablePeerAccess_fn, 38)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceEnablePeerAccess_fn, 39)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGet_fn, 40)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetAttribute_fn, 41)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetByPCIBusId_fn, 42)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetCacheConfig_fn, 43)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetDefaultMemPool_fn, 44)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetGraphMemAttribute_fn, 45)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetLimit_fn, 46)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetMemPool_fn, 47)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetName_fn, 48)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetP2PAttribute_fn, 49)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetPCIBusId_fn, 50)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetSharedMemConfig_fn, 51)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetStreamPriorityRange_fn, 52)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetUuid_fn, 53)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGraphMemTrim_fn, 54)
HIP_ENFORCE_ABI(HipDispatchTable, hipDevicePrimaryCtxGetState_fn, 55)
HIP_ENFORCE_ABI(HipDispatchTable, hipDevicePrimaryCtxRelease_fn, 56)
HIP_ENFORCE_ABI(HipDispatchTable, hipDevicePrimaryCtxReset_fn, 57)
HIP_ENFORCE_ABI(HipDispatchTable, hipDevicePrimaryCtxRetain_fn, 58)
HIP_ENFORCE_ABI(HipDispatchTable, hipDevicePrimaryCtxSetFlags_fn, 59)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceReset_fn, 60)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceSetCacheConfig_fn, 61)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceSetGraphMemAttribute_fn, 62)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceSetLimit_fn, 63)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceSetMemPool_fn, 64)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceSetSharedMemConfig_fn, 65)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceSynchronize_fn, 66)
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceTotalMem_fn, 67)
HIP_ENFORCE_ABI(HipDispatchTable, hipDriverGetVersion_fn, 68)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGetErrorName_fn, 69)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGetErrorString_fn, 70)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGraphAddMemcpyNode_fn, 71)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvMemcpy2DUnaligned_fn, 72)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvMemcpy3D_fn, 73)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvMemcpy3DAsync_fn, 74)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvPointerGetAttributes_fn, 75)
HIP_ENFORCE_ABI(HipDispatchTable, hipEventCreate_fn, 76)
HIP_ENFORCE_ABI(HipDispatchTable, hipEventCreateWithFlags_fn, 77)
HIP_ENFORCE_ABI(HipDispatchTable, hipEventDestroy_fn, 78)
HIP_ENFORCE_ABI(HipDispatchTable, hipEventElapsedTime_fn, 79)
HIP_ENFORCE_ABI(HipDispatchTable, hipEventQuery_fn, 80)
HIP_ENFORCE_ABI(HipDispatchTable, hipEventRecord_fn, 81)
HIP_ENFORCE_ABI(HipDispatchTable, hipEventSynchronize_fn, 82)
HIP_ENFORCE_ABI(HipDispatchTable, hipExtGetLinkTypeAndHopCount_fn, 83)
HIP_ENFORCE_ABI(HipDispatchTable, hipExtLaunchKernel_fn, 84)
HIP_ENFORCE_ABI(HipDispatchTable, hipExtLaunchMultiKernelMultiDevice_fn, 85)
HIP_ENFORCE_ABI(HipDispatchTable, hipExtMallocWithFlags_fn, 86)
HIP_ENFORCE_ABI(HipDispatchTable, hipExtStreamCreateWithCUMask_fn, 87)
HIP_ENFORCE_ABI(HipDispatchTable, hipExtStreamGetCUMask_fn, 88)
HIP_ENFORCE_ABI(HipDispatchTable, hipExternalMemoryGetMappedBuffer_fn, 89)
HIP_ENFORCE_ABI(HipDispatchTable, hipFree_fn, 90)
HIP_ENFORCE_ABI(HipDispatchTable, hipFreeArray_fn, 91)
HIP_ENFORCE_ABI(HipDispatchTable, hipFreeAsync_fn, 92)
HIP_ENFORCE_ABI(HipDispatchTable, hipFreeHost_fn, 93)
HIP_ENFORCE_ABI(HipDispatchTable, hipFreeMipmappedArray_fn, 94)
HIP_ENFORCE_ABI(HipDispatchTable, hipFuncGetAttribute_fn, 95)
HIP_ENFORCE_ABI(HipDispatchTable, hipFuncGetAttributes_fn, 96)
HIP_ENFORCE_ABI(HipDispatchTable, hipFuncSetAttribute_fn, 97)
HIP_ENFORCE_ABI(HipDispatchTable, hipFuncSetCacheConfig_fn, 98)
HIP_ENFORCE_ABI(HipDispatchTable, hipFuncSetSharedMemConfig_fn, 99)
HIP_ENFORCE_ABI(HipDispatchTable, hipGLGetDevices_fn, 100)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetChannelDesc_fn, 101)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetDevice_fn, 102)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetDeviceCount_fn, 103)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetDeviceFlags_fn, 104)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetDevicePropertiesR0600_fn, 105)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetDevicePropertiesR0000_fn, 106)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetErrorName_fn, 107)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetErrorString_fn, 108)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetLastError_fn, 109)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetMipmappedArrayLevel_fn, 110)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetSymbolAddress_fn, 111)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetSymbolSize_fn, 112)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetTextureAlignmentOffset_fn, 113)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetTextureObjectResourceDesc_fn, 114)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetTextureObjectResourceViewDesc_fn, 115)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetTextureObjectTextureDesc_fn, 116)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetTextureReference_fn, 117)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddChildGraphNode_fn, 118)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddDependencies_fn, 119)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddEmptyNode_fn, 120)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddEventRecordNode_fn, 121)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddEventWaitNode_fn, 122)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddHostNode_fn, 123)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddKernelNode_fn, 124)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddMemAllocNode_fn, 125)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddMemFreeNode_fn, 126)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddMemcpyNode_fn, 127)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddMemcpyNode1D_fn, 128)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddMemcpyNodeFromSymbol_fn, 129)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddMemcpyNodeToSymbol_fn, 130)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddMemsetNode_fn, 131)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphChildGraphNodeGetGraph_fn, 132)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphClone_fn, 133)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphCreate_fn, 134)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphDebugDotPrint_fn, 135)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphDestroy_fn, 136)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphDestroyNode_fn, 137)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphEventRecordNodeGetEvent_fn, 138)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphEventRecordNodeSetEvent_fn, 139)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphEventWaitNodeGetEvent_fn, 140)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphEventWaitNodeSetEvent_fn, 141)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecChildGraphNodeSetParams_fn, 142)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecDestroy_fn, 143)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecEventRecordNodeSetEvent_fn, 144)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecEventWaitNodeSetEvent_fn, 145)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecHostNodeSetParams_fn, 146)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecKernelNodeSetParams_fn, 147)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecMemcpyNodeSetParams_fn, 148)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecMemcpyNodeSetParams1D_fn, 149)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecMemcpyNodeSetParamsFromSymbol_fn, 150)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecMemcpyNodeSetParamsToSymbol_fn, 151)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecMemsetNodeSetParams_fn, 152)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecUpdate_fn, 153)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphGetEdges_fn, 154)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphGetNodes_fn, 155)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphGetRootNodes_fn, 156)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphHostNodeGetParams_fn, 157)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphHostNodeSetParams_fn, 158)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphInstantiate_fn, 159)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphInstantiateWithFlags_fn, 160)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphKernelNodeCopyAttributes_fn, 161)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphKernelNodeGetAttribute_fn, 162)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphKernelNodeGetParams_fn, 163)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphKernelNodeSetAttribute_fn, 164)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphKernelNodeSetParams_fn, 165)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphLaunch_fn, 166)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemAllocNodeGetParams_fn, 167)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemFreeNodeGetParams_fn, 168)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemcpyNodeGetParams_fn, 169)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemcpyNodeSetParams_fn, 170)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemcpyNodeSetParams1D_fn, 171)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemcpyNodeSetParamsFromSymbol_fn, 172)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemcpyNodeSetParamsToSymbol_fn, 173)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemsetNodeGetParams_fn, 174)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphMemsetNodeSetParams_fn, 175)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphNodeFindInClone_fn, 176)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphNodeGetDependencies_fn, 177)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphNodeGetDependentNodes_fn, 178)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphNodeGetEnabled_fn, 179)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphNodeGetType_fn, 180)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphNodeSetEnabled_fn, 181)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphReleaseUserObject_fn, 182)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphRemoveDependencies_fn, 183)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphRetainUserObject_fn, 184)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphUpload_fn, 185)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphicsGLRegisterBuffer_fn, 186)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphicsGLRegisterImage_fn, 187)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphicsMapResources_fn, 188)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphicsResourceGetMappedPointer_fn, 189)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphicsSubResourceGetMappedArray_fn, 190)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphicsUnmapResources_fn, 191)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphicsUnregisterResource_fn, 192)
HIP_ENFORCE_ABI(HipDispatchTable, hipHostAlloc_fn, 193)
HIP_ENFORCE_ABI(HipDispatchTable, hipHostFree_fn, 194)
HIP_ENFORCE_ABI(HipDispatchTable, hipHostGetDevicePointer_fn, 195)
HIP_ENFORCE_ABI(HipDispatchTable, hipHostGetFlags_fn, 196)
HIP_ENFORCE_ABI(HipDispatchTable, hipHostMalloc_fn, 197)
HIP_ENFORCE_ABI(HipDispatchTable, hipHostRegister_fn, 198)
HIP_ENFORCE_ABI(HipDispatchTable, hipHostUnregister_fn, 199)
HIP_ENFORCE_ABI(HipDispatchTable, hipImportExternalMemory_fn, 200)
HIP_ENFORCE_ABI(HipDispatchTable, hipImportExternalSemaphore_fn, 201)
HIP_ENFORCE_ABI(HipDispatchTable, hipInit_fn, 202)
HIP_ENFORCE_ABI(HipDispatchTable, hipIpcCloseMemHandle_fn, 203)
HIP_ENFORCE_ABI(HipDispatchTable, hipIpcGetEventHandle_fn, 204)
HIP_ENFORCE_ABI(HipDispatchTable, hipIpcGetMemHandle_fn, 205)
HIP_ENFORCE_ABI(HipDispatchTable, hipIpcOpenEventHandle_fn, 206)
HIP_ENFORCE_ABI(HipDispatchTable, hipIpcOpenMemHandle_fn, 207)
HIP_ENFORCE_ABI(HipDispatchTable, hipKernelNameRef_fn, 208)
HIP_ENFORCE_ABI(HipDispatchTable, hipKernelNameRefByPtr_fn, 209)
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchByPtr_fn, 210)
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchCooperativeKernel_fn, 211)
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchCooperativeKernelMultiDevice_fn, 212)
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchHostFunc_fn, 213)
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchKernel_fn, 214)
HIP_ENFORCE_ABI(HipDispatchTable, hipMalloc_fn, 215)
HIP_ENFORCE_ABI(HipDispatchTable, hipMalloc3D_fn, 216)
HIP_ENFORCE_ABI(HipDispatchTable, hipMalloc3DArray_fn, 217)
HIP_ENFORCE_ABI(HipDispatchTable, hipMallocArray_fn, 218)
HIP_ENFORCE_ABI(HipDispatchTable, hipMallocAsync_fn, 219)
HIP_ENFORCE_ABI(HipDispatchTable, hipMallocFromPoolAsync_fn, 220)
HIP_ENFORCE_ABI(HipDispatchTable, hipMallocHost_fn, 221)
HIP_ENFORCE_ABI(HipDispatchTable, hipMallocManaged_fn, 222)
HIP_ENFORCE_ABI(HipDispatchTable, hipMallocMipmappedArray_fn, 223)
HIP_ENFORCE_ABI(HipDispatchTable, hipMallocPitch_fn, 224)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemAddressFree_fn, 225)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemAddressReserve_fn, 226)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemAdvise_fn, 227)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemAllocHost_fn, 228)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemAllocPitch_fn, 229)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemCreate_fn, 230)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemExportToShareableHandle_fn, 231)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemGetAccess_fn, 232)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemGetAddressRange_fn, 233)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemGetAllocationGranularity_fn, 234)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemGetAllocationPropertiesFromHandle_fn, 235)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemGetInfo_fn, 236)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemImportFromShareableHandle_fn, 237)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemMap_fn, 238)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemMapArrayAsync_fn, 239)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolCreate_fn, 240)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolDestroy_fn, 241)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolExportPointer_fn, 242)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolExportToShareableHandle_fn, 243)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolGetAccess_fn, 244)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolGetAttribute_fn, 245)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolImportFromShareableHandle_fn, 246)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolImportPointer_fn, 247)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolSetAccess_fn, 248)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolSetAttribute_fn, 249)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPoolTrimTo_fn, 250)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPrefetchAsync_fn, 251)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPtrGetInfo_fn, 252)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemRangeGetAttribute_fn, 253)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemRangeGetAttributes_fn, 254)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemRelease_fn, 255)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemRetainAllocationHandle_fn, 256)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemSetAccess_fn, 257)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemUnmap_fn, 258)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy_fn, 259)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2D_fn, 260)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DAsync_fn, 261)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DFromArray_fn, 262)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DFromArrayAsync_fn, 263)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DToArray_fn, 264)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DToArrayAsync_fn, 265)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy3D_fn, 266)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy3DAsync_fn, 267)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyAsync_fn, 268)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyAtoH_fn, 269)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyDtoD_fn, 270)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyDtoDAsync_fn, 271)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyDtoH_fn, 272)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyDtoHAsync_fn, 273)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyFromArray_fn, 274)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyFromSymbol_fn, 275)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyFromSymbolAsync_fn, 276)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyHtoA_fn, 277)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyHtoD_fn, 278)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyHtoDAsync_fn, 279)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyParam2D_fn, 280)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyParam2DAsync_fn, 281)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyPeer_fn, 282)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyPeerAsync_fn, 283)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyToArray_fn, 284)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyToSymbol_fn, 285)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyToSymbolAsync_fn, 286)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyWithStream_fn, 287)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset_fn, 288)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset2D_fn, 289)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset2DAsync_fn, 290)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset3D_fn, 291)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset3DAsync_fn, 292)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetAsync_fn, 293)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD16_fn, 294)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD16Async_fn, 295)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD32_fn, 296)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD32Async_fn, 297)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD8_fn, 298)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD8Async_fn, 299)
HIP_ENFORCE_ABI(HipDispatchTable, hipMipmappedArrayCreate_fn, 300)
HIP_ENFORCE_ABI(HipDispatchTable, hipMipmappedArrayDestroy_fn, 301)
HIP_ENFORCE_ABI(HipDispatchTable, hipMipmappedArrayGetLevel_fn, 302)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleGetFunction_fn, 303)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleGetGlobal_fn, 304)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleGetTexRef_fn, 305)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleLaunchCooperativeKernel_fn, 306)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleLaunchCooperativeKernelMultiDevice_fn, 307)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleLaunchKernel_fn, 308)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleLoad_fn, 309)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleLoadData_fn, 310)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleLoadDataEx_fn, 311)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn, 312)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn,
                313)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleOccupancyMaxPotentialBlockSize_fn, 314)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn, 315)
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleUnload_fn, 316)
HIP_ENFORCE_ABI(HipDispatchTable, hipOccupancyMaxActiveBlocksPerMultiprocessor_fn, 317)
HIP_ENFORCE_ABI(HipDispatchTable, hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn, 318)
HIP_ENFORCE_ABI(HipDispatchTable, hipOccupancyMaxPotentialBlockSize_fn, 319)
HIP_ENFORCE_ABI(HipDispatchTable, hipPeekAtLastError_fn, 320)
HIP_ENFORCE_ABI(HipDispatchTable, hipPointerGetAttribute_fn, 321)
HIP_ENFORCE_ABI(HipDispatchTable, hipPointerGetAttributes_fn, 322)
HIP_ENFORCE_ABI(HipDispatchTable, hipPointerSetAttribute_fn, 323)
HIP_ENFORCE_ABI(HipDispatchTable, hipProfilerStart_fn, 324)
HIP_ENFORCE_ABI(HipDispatchTable, hipProfilerStop_fn, 325)
HIP_ENFORCE_ABI(HipDispatchTable, hipRuntimeGetVersion_fn, 326)
HIP_ENFORCE_ABI(HipDispatchTable, hipSetDevice_fn, 327)
HIP_ENFORCE_ABI(HipDispatchTable, hipSetDeviceFlags_fn, 328)
HIP_ENFORCE_ABI(HipDispatchTable, hipSetupArgument_fn, 329)
HIP_ENFORCE_ABI(HipDispatchTable, hipSignalExternalSemaphoresAsync_fn, 330)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamAddCallback_fn, 331)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamAttachMemAsync_fn, 332)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamBeginCapture_fn, 333)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamCreate_fn, 334)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamCreateWithFlags_fn, 335)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamCreateWithPriority_fn, 336)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamDestroy_fn, 337)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamEndCapture_fn, 338)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetCaptureInfo_fn, 339)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetCaptureInfo_v2_fn, 340)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetDevice_fn, 341)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetFlags_fn, 342)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetPriority_fn, 343)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamIsCapturing_fn, 344)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamQuery_fn, 345)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamSynchronize_fn, 346)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamUpdateCaptureDependencies_fn, 347)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamWaitEvent_fn, 348)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamWaitValue32_fn, 349)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamWaitValue64_fn, 350)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamWriteValue32_fn, 351)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamWriteValue64_fn, 352)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexObjectCreate_fn, 353)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexObjectDestroy_fn, 354)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexObjectGetResourceDesc_fn, 355)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexObjectGetResourceViewDesc_fn, 356)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexObjectGetTextureDesc_fn, 357)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetAddress_fn, 358)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetAddressMode_fn, 359)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetFilterMode_fn, 360)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetFlags_fn, 361)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetFormat_fn, 362)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetMaxAnisotropy_fn, 363)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetMipMappedArray_fn, 364)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetMipmapFilterMode_fn, 365)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetMipmapLevelBias_fn, 366)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetMipmapLevelClamp_fn, 367)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetAddress_fn, 368)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetAddress2D_fn, 369)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetAddressMode_fn, 370)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetArray_fn, 371)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetBorderColor_fn, 372)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetFilterMode_fn, 373)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetFlags_fn, 374)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetFormat_fn, 375)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetMaxAnisotropy_fn, 376)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetMipmapFilterMode_fn, 377)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetMipmapLevelBias_fn, 378)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetMipmapLevelClamp_fn, 379)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefSetMipmappedArray_fn, 380)
HIP_ENFORCE_ABI(HipDispatchTable, hipThreadExchangeStreamCaptureMode_fn, 381)
HIP_ENFORCE_ABI(HipDispatchTable, hipUnbindTexture_fn, 382)
HIP_ENFORCE_ABI(HipDispatchTable, hipUserObjectCreate_fn, 383)
HIP_ENFORCE_ABI(HipDispatchTable, hipUserObjectRelease_fn, 384)
HIP_ENFORCE_ABI(HipDispatchTable, hipUserObjectRetain_fn, 385)
HIP_ENFORCE_ABI(HipDispatchTable, hipWaitExternalSemaphoresAsync_fn, 386)
HIP_ENFORCE_ABI(HipDispatchTable, hipCreateChannelDesc_fn, 387)
HIP_ENFORCE_ABI(HipDispatchTable, hipExtModuleLaunchKernel_fn, 388)
HIP_ENFORCE_ABI(HipDispatchTable, hipHccModuleLaunchKernel_fn, 389)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy_spt_fn, 390)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyToSymbol_spt_fn, 391)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyFromSymbol_spt_fn, 392)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2D_spt_fn, 393)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DFromArray_spt_fn, 394)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy3D_spt_fn, 395)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset_spt_fn, 396)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetAsync_spt_fn, 397)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset2D_spt_fn, 398)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset2DAsync_spt_fn, 399)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset3DAsync_spt_fn, 400)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemset3D_spt_fn, 401)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyAsync_spt_fn, 402)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy3DAsync_spt_fn, 403)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DAsync_spt_fn, 404)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyFromSymbolAsync_spt_fn, 405)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyToSymbolAsync_spt_fn, 406)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyFromArray_spt_fn, 407)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DToArray_spt_fn, 408)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DFromArrayAsync_spt_fn, 409)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DToArrayAsync_spt_fn, 410)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamQuery_spt_fn, 411)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamSynchronize_spt_fn, 412)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetPriority_spt_fn, 413)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamWaitEvent_spt_fn, 414)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetFlags_spt_fn, 415)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamAddCallback_spt_fn, 416)
HIP_ENFORCE_ABI(HipDispatchTable, hipEventRecord_spt_fn, 417)
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchCooperativeKernel_spt_fn, 418)
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchKernel_spt_fn, 419)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphLaunch_spt_fn, 420)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamBeginCapture_spt_fn, 421)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamEndCapture_spt_fn, 422)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamIsCapturing_spt_fn, 423)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetCaptureInfo_spt_fn, 424)
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetCaptureInfo_v2_spt_fn, 425)
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchHostFunc_spt_fn, 426)
HIP_ENFORCE_ABI(HipDispatchTable, hipGetStreamDeviceId_fn, 427)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGraphAddMemsetNode_fn, 428)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddExternalSemaphoresWaitNode_fn, 429)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddExternalSemaphoresSignalNode_fn, 430)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExternalSemaphoresSignalNodeSetParams_fn, 431)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExternalSemaphoresWaitNodeSetParams_fn, 432)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExternalSemaphoresSignalNodeGetParams_fn, 433)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExternalSemaphoresWaitNodeGetParams_fn, 434)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecExternalSemaphoresSignalNodeSetParams_fn, 435)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecExternalSemaphoresWaitNodeSetParams_fn, 436)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddNode_fn, 437)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphInstantiateWithParams_fn, 438)
HIP_ENFORCE_ABI(HipDispatchTable, hipExtGetLastError_fn, 439)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetBorderColor_fn, 440)
HIP_ENFORCE_ABI(HipDispatchTable, hipTexRefGetArray_fn, 441)
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 1
HIP_ENFORCE_ABI(HipDispatchTable, hipGetProcAddress_fn, 442)
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 2
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamBeginCaptureToGraph_fn, 443)
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 3
HIP_ENFORCE_ABI(HipDispatchTable, hipGetFuncBySymbol_fn, 444)
HIP_ENFORCE_ABI(HipDispatchTable, hipSetValidDevices_fn, 445)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyAtoD_fn, 446)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyDtoA_fn, 447)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyAtoA_fn, 448)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyAtoHAsync_fn, 449)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyHtoAAsync_fn, 450)
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy2DArrayToArray_fn, 451)
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 4
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGraphAddMemFreeNode_fn, 452)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGraphExecMemcpyNodeSetParams_fn, 453)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGraphExecMemsetNodeSetParams_fn, 454)
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecGetFlags_fn, 455);
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphNodeSetParams_fn, 456);
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecNodeSetParams_fn, 457);
HIP_ENFORCE_ABI(HipDispatchTable, hipExternalMemoryGetMappedMipmappedArray_fn, 458)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGraphMemcpyNodeGetParams_fn, 459)
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvGraphMemcpyNodeSetParams_fn, 460)
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 5
HIP_ENFORCE_ABI(HipDispatchTable, hipExtHostAlloc_fn, 461)
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 6
HIP_ENFORCE_ABI(HipDispatchTable, hipDeviceGetTexture1DLinearMaxWidth_fn, 462)
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 7
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamBatchMemOp_fn, 463);
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 8
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphAddBatchMemOpNode_fn, 464);
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphBatchMemOpNodeGetParams_fn, 465);
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphBatchMemOpNodeSetParams_fn, 466);
HIP_ENFORCE_ABI(HipDispatchTable, hipGraphExecBatchMemOpNodeSetParams_fn, 467);
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 9
HIP_ENFORCE_ABI(HipDispatchTable, hipLinkAddData_fn, 468)
HIP_ENFORCE_ABI(HipDispatchTable, hipLinkAddFile_fn, 469)
HIP_ENFORCE_ABI(HipDispatchTable, hipLinkComplete_fn, 470)
HIP_ENFORCE_ABI(HipDispatchTable, hipLinkCreate_fn, 471)
HIP_ENFORCE_ABI(HipDispatchTable, hipLinkDestroy_fn, 472)
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 10
HIP_ENFORCE_ABI(HipDispatchTable, hipEventRecordWithFlags_fn, 473)

// HIP_RUNTIME_API_TABLE_STEP_VERSION == 11
HIP_ENFORCE_ABI(HipDispatchTable, hipLaunchKernelExC_fn, 474);
HIP_ENFORCE_ABI(HipDispatchTable, hipDrvLaunchKernelEx_fn, 475);
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 12
HIP_ENFORCE_ABI(HipDispatchTable, hipMemGetHandleForAddressRange_fn, 476);
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 13
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 14
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleGetFunctionCount_fn, 477);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD2D8_fn, 478);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD2D8Async_fn, 479);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD2D16_fn, 480);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD2D16Async_fn, 481);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD2D32_fn, 482);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemsetD2D32Async_fn, 483);
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetAttribute_fn, 484);
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamSetAttribute_fn, 485);
HIP_ENFORCE_ABI(HipDispatchTable, hipModuleLoadFatBinary_fn, 486);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpyBatchAsync_fn, 487);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy3DBatchAsync_fn, 488);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy3DPeer_fn, 489);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemcpy3DPeerAsync_fn, 490);
HIP_ENFORCE_ABI(HipDispatchTable, hipGetDriverEntryPoint_fn, 491);
HIP_ENFORCE_ABI(HipDispatchTable, hipGetDriverEntryPoint_spt_fn, 492);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemPrefetchAsync_v2_fn, 493);
HIP_ENFORCE_ABI(HipDispatchTable, hipMemAdvise_v2_fn, 494);
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamGetId_fn, 495);
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 15
HIP_ENFORCE_ABI(HipDispatchTable, hipLibraryLoadData_fn, 496);
HIP_ENFORCE_ABI(HipDispatchTable, hipLibraryLoadFromFile_fn, 497);
HIP_ENFORCE_ABI(HipDispatchTable, hipLibraryUnload_fn, 498);
HIP_ENFORCE_ABI(HipDispatchTable, hipLibraryGetKernel_fn, 499);
HIP_ENFORCE_ABI(HipDispatchTable, hipLibraryGetKernelCount_fn, 500);
// HIP_RUNTIME_API_TABLE_STEP_VERSION == 16
HIP_ENFORCE_ABI(HipDispatchTable, hipStreamCopyAttributes_fn, 501);
// if HIP_ENFORCE_ABI entries are added for each new function pointer in the table, the number below
// will be +1 of the number in the last HIP_ENFORCE_ABI line. E.g.:
//
//  HIP_ENFORCE_ABI(<table>, <functor>, 8)
//
//  HIP_ENFORCE_ABI_VERSIONING(<table>, 9) <- 8 + 1 = 9
HIP_ENFORCE_ABI_VERSIONING(HipDispatchTable, 502)

static_assert(HIP_RUNTIME_API_TABLE_MAJOR_VERSION == 0 && HIP_RUNTIME_API_TABLE_STEP_VERSION == 16,
              "If you get this error, add new HIP_ENFORCE_ABI(...) code for the new function "
              "pointers and then update this check so it is true");
#endif
