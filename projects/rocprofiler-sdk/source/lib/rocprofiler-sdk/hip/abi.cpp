// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/hip.h>

#include "lib/common/abi.hpp"
#include "lib/common/defines.hpp"

namespace rocprofiler
{
namespace hip
{
static_assert(HIP_COMPILER_API_TABLE_MAJOR_VERSION == 0,
              "Major version updated for HIP compiler dispatch table");
static_assert(HIP_RUNTIME_API_TABLE_MAJOR_VERSION == 0,
              "Major version updated for HIP runtime dispatch table");

// These ensure that function pointers are not re-ordered
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipPopCallConfiguration_fn, 0)
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipPushCallConfiguration_fn, 1)
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipRegisterFatBinary_fn, 2)
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipRegisterFunction_fn, 3)
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipRegisterManagedVar_fn, 4)
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipRegisterSurface_fn, 5)
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipRegisterTexture_fn, 6)
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipRegisterVar_fn, 7)
ROCP_SDK_ENFORCE_ABI(::HipCompilerDispatchTable, __hipUnregisterFatBinary_fn, 8)

#if HIP_COMPILER_API_TABLE_STEP_VERSION == 0
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipCompilerDispatchTable, 9)
#else
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipCompilerDispatchTable, 0)
#endif

// These ensure that function pointers are not re-ordered
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipApiName_fn, 0)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipArray3DCreate_fn, 1)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipArray3DGetDescriptor_fn, 2)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipArrayCreate_fn, 3)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipArrayDestroy_fn, 4)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipArrayGetDescriptor_fn, 5)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipArrayGetInfo_fn, 6)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipBindTexture_fn, 7)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipBindTexture2D_fn, 8)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipBindTextureToArray_fn, 9)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipBindTextureToMipmappedArray_fn, 10)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipChooseDevice_fn, 11)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipChooseDeviceR0000_fn, 12)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipConfigureCall_fn, 13)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCreateSurfaceObject_fn, 14)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCreateTextureObject_fn, 15)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxCreate_fn, 16)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxDestroy_fn, 17)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxDisablePeerAccess_fn, 18)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxEnablePeerAccess_fn, 19)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxGetApiVersion_fn, 20)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxGetCacheConfig_fn, 21)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxGetCurrent_fn, 22)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxGetDevice_fn, 23)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxGetFlags_fn, 24)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxGetSharedMemConfig_fn, 25)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxPopCurrent_fn, 26)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxPushCurrent_fn, 27)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxSetCacheConfig_fn, 28)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxSetCurrent_fn, 29)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxSetSharedMemConfig_fn, 30)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCtxSynchronize_fn, 31)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDestroyExternalMemory_fn, 32)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDestroyExternalSemaphore_fn, 33)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDestroySurfaceObject_fn, 34)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDestroyTextureObject_fn, 35)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceCanAccessPeer_fn, 36)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceComputeCapability_fn, 37)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceDisablePeerAccess_fn, 38)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceEnablePeerAccess_fn, 39)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGet_fn, 40)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetAttribute_fn, 41)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetByPCIBusId_fn, 42)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetCacheConfig_fn, 43)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetDefaultMemPool_fn, 44)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetGraphMemAttribute_fn, 45)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetLimit_fn, 46)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetMemPool_fn, 47)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetName_fn, 48)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetP2PAttribute_fn, 49)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetPCIBusId_fn, 50)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetSharedMemConfig_fn, 51)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetStreamPriorityRange_fn, 52)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetUuid_fn, 53)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGraphMemTrim_fn, 54)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDevicePrimaryCtxGetState_fn, 55)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDevicePrimaryCtxRelease_fn, 56)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDevicePrimaryCtxReset_fn, 57)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDevicePrimaryCtxRetain_fn, 58)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDevicePrimaryCtxSetFlags_fn, 59)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceReset_fn, 60)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceSetCacheConfig_fn, 61)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceSetGraphMemAttribute_fn, 62)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceSetLimit_fn, 63)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceSetMemPool_fn, 64)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceSetSharedMemConfig_fn, 65)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceSynchronize_fn, 66)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceTotalMem_fn, 67)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDriverGetVersion_fn, 68)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGetErrorName_fn, 69)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGetErrorString_fn, 70)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGraphAddMemcpyNode_fn, 71)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvMemcpy2DUnaligned_fn, 72)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvMemcpy3D_fn, 73)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvMemcpy3DAsync_fn, 74)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvPointerGetAttributes_fn, 75)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventCreate_fn, 76)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventCreateWithFlags_fn, 77)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventDestroy_fn, 78)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventElapsedTime_fn, 79)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventQuery_fn, 80)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventRecord_fn, 81)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventSynchronize_fn, 82)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtGetLinkTypeAndHopCount_fn, 83)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtLaunchKernel_fn, 84)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtLaunchMultiKernelMultiDevice_fn, 85)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtMallocWithFlags_fn, 86)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtStreamCreateWithCUMask_fn, 87)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtStreamGetCUMask_fn, 88)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExternalMemoryGetMappedBuffer_fn, 89)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFree_fn, 90)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFreeArray_fn, 91)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFreeAsync_fn, 92)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFreeHost_fn, 93)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFreeMipmappedArray_fn, 94)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFuncGetAttribute_fn, 95)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFuncGetAttributes_fn, 96)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFuncSetAttribute_fn, 97)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFuncSetCacheConfig_fn, 98)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipFuncSetSharedMemConfig_fn, 99)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGLGetDevices_fn, 100)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetChannelDesc_fn, 101)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetDevice_fn, 102)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetDeviceCount_fn, 103)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetDeviceFlags_fn, 104)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetDevicePropertiesR0600_fn, 105)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetDevicePropertiesR0000_fn, 106)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetErrorName_fn, 107)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetErrorString_fn, 108)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetLastError_fn, 109)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetMipmappedArrayLevel_fn, 110)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetSymbolAddress_fn, 111)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetSymbolSize_fn, 112)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetTextureAlignmentOffset_fn, 113)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetTextureObjectResourceDesc_fn, 114)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetTextureObjectResourceViewDesc_fn, 115)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetTextureObjectTextureDesc_fn, 116)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetTextureReference_fn, 117)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddChildGraphNode_fn, 118)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddDependencies_fn, 119)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddEmptyNode_fn, 120)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddEventRecordNode_fn, 121)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddEventWaitNode_fn, 122)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddHostNode_fn, 123)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddKernelNode_fn, 124)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddMemAllocNode_fn, 125)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddMemFreeNode_fn, 126)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddMemcpyNode_fn, 127)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddMemcpyNode1D_fn, 128)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddMemcpyNodeFromSymbol_fn, 129)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddMemcpyNodeToSymbol_fn, 130)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddMemsetNode_fn, 131)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphChildGraphNodeGetGraph_fn, 132)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphClone_fn, 133)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphCreate_fn, 134)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphDebugDotPrint_fn, 135)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphDestroy_fn, 136)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphDestroyNode_fn, 137)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphEventRecordNodeGetEvent_fn, 138)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphEventRecordNodeSetEvent_fn, 139)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphEventWaitNodeGetEvent_fn, 140)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphEventWaitNodeSetEvent_fn, 141)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecChildGraphNodeSetParams_fn, 142)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecDestroy_fn, 143)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecEventRecordNodeSetEvent_fn, 144)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecEventWaitNodeSetEvent_fn, 145)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecHostNodeSetParams_fn, 146)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecKernelNodeSetParams_fn, 147)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecMemcpyNodeSetParams_fn, 148)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecMemcpyNodeSetParams1D_fn, 149)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecMemcpyNodeSetParamsFromSymbol_fn, 150)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecMemcpyNodeSetParamsToSymbol_fn, 151)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecMemsetNodeSetParams_fn, 152)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecUpdate_fn, 153)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphGetEdges_fn, 154)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphGetNodes_fn, 155)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphGetRootNodes_fn, 156)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphHostNodeGetParams_fn, 157)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphHostNodeSetParams_fn, 158)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphInstantiate_fn, 159)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphInstantiateWithFlags_fn, 160)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphKernelNodeCopyAttributes_fn, 161)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphKernelNodeGetAttribute_fn, 162)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphKernelNodeGetParams_fn, 163)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphKernelNodeSetAttribute_fn, 164)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphKernelNodeSetParams_fn, 165)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphLaunch_fn, 166)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemAllocNodeGetParams_fn, 167)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemFreeNodeGetParams_fn, 168)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemcpyNodeGetParams_fn, 169)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemcpyNodeSetParams_fn, 170)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemcpyNodeSetParams1D_fn, 171)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemcpyNodeSetParamsFromSymbol_fn, 172)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemcpyNodeSetParamsToSymbol_fn, 173)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemsetNodeGetParams_fn, 174)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphMemsetNodeSetParams_fn, 175)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphNodeFindInClone_fn, 176)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphNodeGetDependencies_fn, 177)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphNodeGetDependentNodes_fn, 178)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphNodeGetEnabled_fn, 179)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphNodeGetType_fn, 180)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphNodeSetEnabled_fn, 181)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphReleaseUserObject_fn, 182)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphRemoveDependencies_fn, 183)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphRetainUserObject_fn, 184)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphUpload_fn, 185)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphicsGLRegisterBuffer_fn, 186)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphicsGLRegisterImage_fn, 187)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphicsMapResources_fn, 188)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphicsResourceGetMappedPointer_fn, 189)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphicsSubResourceGetMappedArray_fn, 190)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphicsUnmapResources_fn, 191)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphicsUnregisterResource_fn, 192)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipHostAlloc_fn, 193)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipHostFree_fn, 194)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipHostGetDevicePointer_fn, 195)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipHostGetFlags_fn, 196)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipHostMalloc_fn, 197)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipHostRegister_fn, 198)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipHostUnregister_fn, 199)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipImportExternalMemory_fn, 200)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipImportExternalSemaphore_fn, 201)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipInit_fn, 202)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipIpcCloseMemHandle_fn, 203)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipIpcGetEventHandle_fn, 204)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipIpcGetMemHandle_fn, 205)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipIpcOpenEventHandle_fn, 206)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipIpcOpenMemHandle_fn, 207)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipKernelNameRef_fn, 208)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipKernelNameRefByPtr_fn, 209)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchByPtr_fn, 210)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchCooperativeKernel_fn, 211)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchCooperativeKernelMultiDevice_fn, 212)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchHostFunc_fn, 213)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchKernel_fn, 214)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMalloc_fn, 215)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMalloc3D_fn, 216)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMalloc3DArray_fn, 217)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMallocArray_fn, 218)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMallocAsync_fn, 219)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMallocFromPoolAsync_fn, 220)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMallocHost_fn, 221)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMallocManaged_fn, 222)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMallocMipmappedArray_fn, 223)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMallocPitch_fn, 224)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemAddressFree_fn, 225)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemAddressReserve_fn, 226)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemAdvise_fn, 227)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemAllocHost_fn, 228)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemAllocPitch_fn, 229)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemCreate_fn, 230)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemExportToShareableHandle_fn, 231)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemGetAccess_fn, 232)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemGetAddressRange_fn, 233)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemGetAllocationGranularity_fn, 234)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemGetAllocationPropertiesFromHandle_fn, 235)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemGetInfo_fn, 236)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemImportFromShareableHandle_fn, 237)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemMap_fn, 238)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemMapArrayAsync_fn, 239)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolCreate_fn, 240)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolDestroy_fn, 241)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolExportPointer_fn, 242)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolExportToShareableHandle_fn, 243)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolGetAccess_fn, 244)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolGetAttribute_fn, 245)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolImportFromShareableHandle_fn, 246)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolImportPointer_fn, 247)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolSetAccess_fn, 248)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolSetAttribute_fn, 249)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPoolTrimTo_fn, 250)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPrefetchAsync_fn, 251)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemPtrGetInfo_fn, 252)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemRangeGetAttribute_fn, 253)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemRangeGetAttributes_fn, 254)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemRelease_fn, 255)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemRetainAllocationHandle_fn, 256)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemSetAccess_fn, 257)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemUnmap_fn, 258)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy_fn, 259)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2D_fn, 260)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DAsync_fn, 261)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DFromArray_fn, 262)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DFromArrayAsync_fn, 263)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DToArray_fn, 264)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DToArrayAsync_fn, 265)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy3D_fn, 266)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy3DAsync_fn, 267)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyAsync_fn, 268)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyAtoH_fn, 269)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyDtoD_fn, 270)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyDtoDAsync_fn, 271)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyDtoH_fn, 272)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyDtoHAsync_fn, 273)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyFromArray_fn, 274)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyFromSymbol_fn, 275)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyFromSymbolAsync_fn, 276)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyHtoA_fn, 277)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyHtoD_fn, 278)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyHtoDAsync_fn, 279)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyParam2D_fn, 280)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyParam2DAsync_fn, 281)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyPeer_fn, 282)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyPeerAsync_fn, 283)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyToArray_fn, 284)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyToSymbol_fn, 285)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyToSymbolAsync_fn, 286)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyWithStream_fn, 287)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset_fn, 288)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset2D_fn, 289)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset2DAsync_fn, 290)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset3D_fn, 291)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset3DAsync_fn, 292)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemsetAsync_fn, 293)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemsetD16_fn, 294)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemsetD16Async_fn, 295)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemsetD32_fn, 296)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemsetD32Async_fn, 297)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemsetD8_fn, 298)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemsetD8Async_fn, 299)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMipmappedArrayCreate_fn, 300)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMipmappedArrayDestroy_fn, 301)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMipmappedArrayGetLevel_fn, 302)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleGetFunction_fn, 303)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleGetGlobal_fn, 304)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleGetTexRef_fn, 305)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleLaunchCooperativeKernel_fn, 306)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleLaunchCooperativeKernelMultiDevice_fn, 307)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleLaunchKernel_fn, 308)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleLoad_fn, 309)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleLoadData_fn, 310)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleLoadDataEx_fn, 311)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_fn, 312)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable,
                     hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn,
                     313)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleOccupancyMaxPotentialBlockSize_fn, 314)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleOccupancyMaxPotentialBlockSizeWithFlags_fn, 315)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipModuleUnload_fn, 316)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipOccupancyMaxActiveBlocksPerMultiprocessor_fn, 317)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable,
                     hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn,
                     318)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipOccupancyMaxPotentialBlockSize_fn, 319)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipPeekAtLastError_fn, 320)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipPointerGetAttribute_fn, 321)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipPointerGetAttributes_fn, 322)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipPointerSetAttribute_fn, 323)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipProfilerStart_fn, 324)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipProfilerStop_fn, 325)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipRuntimeGetVersion_fn, 326)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipSetDevice_fn, 327)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipSetDeviceFlags_fn, 328)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipSetupArgument_fn, 329)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipSignalExternalSemaphoresAsync_fn, 330)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamAddCallback_fn, 331)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamAttachMemAsync_fn, 332)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamBeginCapture_fn, 333)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamCreate_fn, 334)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamCreateWithFlags_fn, 335)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamCreateWithPriority_fn, 336)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamDestroy_fn, 337)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamEndCapture_fn, 338)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetCaptureInfo_fn, 339)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetCaptureInfo_v2_fn, 340)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetDevice_fn, 341)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetFlags_fn, 342)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetPriority_fn, 343)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamIsCapturing_fn, 344)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamQuery_fn, 345)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamSynchronize_fn, 346)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamUpdateCaptureDependencies_fn, 347)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamWaitEvent_fn, 348)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamWaitValue32_fn, 349)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamWaitValue64_fn, 350)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamWriteValue32_fn, 351)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamWriteValue64_fn, 352)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexObjectCreate_fn, 353)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexObjectDestroy_fn, 354)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexObjectGetResourceDesc_fn, 355)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexObjectGetResourceViewDesc_fn, 356)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexObjectGetTextureDesc_fn, 357)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetAddress_fn, 358)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetAddressMode_fn, 359)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetFilterMode_fn, 360)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetFlags_fn, 361)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetFormat_fn, 362)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetMaxAnisotropy_fn, 363)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetMipMappedArray_fn, 364)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetMipmapFilterMode_fn, 365)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetMipmapLevelBias_fn, 366)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetMipmapLevelClamp_fn, 367)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetAddress_fn, 368)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetAddress2D_fn, 369)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetAddressMode_fn, 370)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetArray_fn, 371)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetBorderColor_fn, 372)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetFilterMode_fn, 373)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetFlags_fn, 374)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetFormat_fn, 375)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetMaxAnisotropy_fn, 376)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetMipmapFilterMode_fn, 377)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetMipmapLevelBias_fn, 378)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetMipmapLevelClamp_fn, 379)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefSetMipmappedArray_fn, 380)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipThreadExchangeStreamCaptureMode_fn, 381)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipUnbindTexture_fn, 382)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipUserObjectCreate_fn, 383)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipUserObjectRelease_fn, 384)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipUserObjectRetain_fn, 385)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipWaitExternalSemaphoresAsync_fn, 386)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipCreateChannelDesc_fn, 387)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtModuleLaunchKernel_fn, 388)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipHccModuleLaunchKernel_fn, 389)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy_spt_fn, 390)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyToSymbol_spt_fn, 391)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyFromSymbol_spt_fn, 392)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2D_spt_fn, 393)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DFromArray_spt_fn, 394)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy3D_spt_fn, 395)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset_spt_fn, 396)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemsetAsync_spt_fn, 397)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset2D_spt_fn, 398)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset2DAsync_spt_fn, 399)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset3DAsync_spt_fn, 400)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemset3D_spt_fn, 401)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyAsync_spt_fn, 402)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy3DAsync_spt_fn, 403)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DAsync_spt_fn, 404)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyFromSymbolAsync_spt_fn, 405)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyToSymbolAsync_spt_fn, 406)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyFromArray_spt_fn, 407)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DToArray_spt_fn, 408)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DFromArrayAsync_spt_fn, 409)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DToArrayAsync_spt_fn, 410)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamQuery_spt_fn, 411)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamSynchronize_spt_fn, 412)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetPriority_spt_fn, 413)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamWaitEvent_spt_fn, 414)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetFlags_spt_fn, 415)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamAddCallback_spt_fn, 416)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventRecord_spt_fn, 417)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchCooperativeKernel_spt_fn, 418)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchKernel_spt_fn, 419)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphLaunch_spt_fn, 420)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamBeginCapture_spt_fn, 421)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamEndCapture_spt_fn, 422)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamIsCapturing_spt_fn, 423)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetCaptureInfo_spt_fn, 424)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamGetCaptureInfo_v2_spt_fn, 425)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchHostFunc_spt_fn, 426)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetStreamDeviceId_fn, 427)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGraphAddMemsetNode_fn, 428)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddExternalSemaphoresWaitNode_fn, 429);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddExternalSemaphoresSignalNode_fn, 430);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExternalSemaphoresSignalNodeSetParams_fn, 431);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExternalSemaphoresWaitNodeSetParams_fn, 432);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExternalSemaphoresSignalNodeGetParams_fn, 433);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExternalSemaphoresWaitNodeGetParams_fn, 434);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecExternalSemaphoresSignalNodeSetParams_fn, 435);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecExternalSemaphoresWaitNodeSetParams_fn, 436);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddNode_fn, 437);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphInstantiateWithParams_fn, 438);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtGetLastError_fn, 439)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetBorderColor_fn, 440)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipTexRefGetArray_fn, 441)

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 1
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetProcAddress_fn, 442)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 2
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamBeginCaptureToGraph_fn, 443);
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 3
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGetFuncBySymbol_fn, 444);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipSetValidDevices_fn, 445);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyAtoD_fn, 446);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyDtoA_fn, 447);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyAtoA_fn, 448);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyAtoHAsync_fn, 449);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpyHtoAAsync_fn, 450);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemcpy2DArrayToArray_fn, 451);
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 4
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGraphAddMemFreeNode_fn, 452)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGraphExecMemcpyNodeSetParams_fn, 453)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGraphExecMemsetNodeSetParams_fn, 454)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecGetFlags_fn, 455);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphNodeSetParams_fn, 456);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecNodeSetParams_fn, 457);
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExternalMemoryGetMappedMipmappedArray_fn, 458)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGraphMemcpyNodeGetParams_fn, 459)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvGraphMemcpyNodeSetParams_fn, 460)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 5
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipExtHostAlloc_fn, 461)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 6
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDeviceGetTexture1DLinearMaxWidth_fn, 462)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 7
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipStreamBatchMemOp_fn, 463)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 8
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphAddBatchMemOpNode_fn, 464)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphBatchMemOpNodeGetParams_fn, 465)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphBatchMemOpNodeSetParams_fn, 466)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipGraphExecBatchMemOpNodeSetParams_fn, 467)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 9
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLinkAddData_fn, 468)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLinkAddFile_fn, 469)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLinkComplete_fn, 470)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLinkCreate_fn, 471)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLinkDestroy_fn, 472)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipEventRecordWithFlags_fn, 473)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 11
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipLaunchKernelExC_fn, 474)
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipDrvLaunchKernelEx_fn, 475)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 12
ROCP_SDK_ENFORCE_ABI(::HipDispatchTable, hipMemGetHandleForAddressRange_fn, 476)
#endif

#if HIP_RUNTIME_API_TABLE_STEP_VERSION == 0
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 442)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 1
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 443)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 2
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 444)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 3
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 452)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 4
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 461)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 5
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 462)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 6
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 463)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 7
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 464)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 8
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 468)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 9
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 473)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 10
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 474)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 11
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 476)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 12
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 477)
#elif HIP_RUNTIME_API_TABLE_STEP_VERSION == 13
ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 477)
#else
INTERNAL_CI_ROCP_SDK_ENFORCE_ABI_VERSIONING(::HipDispatchTable, 0)
#endif
}  // namespace hip
}  // namespace rocprofiler
