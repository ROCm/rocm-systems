# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# FIXME: cjatin
# This is a temp hack to pass it on windows since it has a char limit for
# command line. Eventually we need to use rsp files with ninja on windows.
# But this exposes several problems with hipcc and build system on windows.
# While those are fixed, this hack should allow us to pass windows CI.
set(TEST_SRC
  hipGraphAddEmptyNode.cc
  hipGraphAddDependencies.cc
  hipGraphAddEventRecordNode.cc
  hipGraphAddEventWaitNode.cc
  hipGraph.cc
  hipSimpleGraphWithKernel.cc
  hipGraphAddMemcpyNode.cc
  hipGraphClone.cc
  hipGraphInstantiateWithFlags.cc
  hipGraphInstantiateWithParams.cc
  hipGraphAddHostNode.cc
  hipGraphAddMemcpyNodeFromSymbol_old.cc
  hipGraphAddMemcpyNodeFromSymbol.cc
  hipGraphChildGraphNodeGetGraph.cc
  hipGraphNodeFindInClone.cc
  hipGraphExecHostNodeSetParams.cc
  hipGraphAddMemcpyNodeToSymbol_old.cc
  hipGraphAddMemcpyNodeToSymbol.cc
  hipGraphExecMemsetNodeSetParams.cc
  hipGraphMemcpyNodeSetParamsToSymbol.cc
  hipGraphDestroyNode.cc
  hipGraphGetNodes.cc
  hipGraphGetRootNodes.cc
  hipGraphGetRootNodes_old.cc
  hipGraphHostNodeSetParams.cc
  hipGraphAddMemcpyNode1D.cc
  hipGraphAddChildGraphNode.cc
  hipGraphNodeGetType.cc
  hipGraphExecMemcpyNodeSetParams1D_old.cc
  hipGraphExecMemcpyNodeSetParams1D.cc
  hipGraphGetEdges.cc
  hipGraphRemoveDependencies.cc
  hipGraphInstantiate.cc
  hipGraphExecUpdate.cc
  hipGraphExecEventRecordNodeSetEvent.cc
  hipGraphEventWaitNodeSetEvent.cc
  hipGraphMemsetNodeGetParams.cc
  hipGraphMemsetNodeSetParams.cc
  hipGraphExecMemcpyNodeSetParamsFromSymbol_old.cc
  hipGraphExecMemcpyNodeSetParamsFromSymbol.cc
  hipGraphEventRecordNodeGetEvent.cc
  hipGraphEventRecordNodeSetEvent.cc
  hipGraphEventWaitNodeGetEvent.cc
  hipGraphExecMemcpyNodeSetParams.cc
  hipGraphExecMemcpyNodeSetParams_old.cc
  hipStreamBeginCapture.cc
  hipGraphAddMemcpyNode1D_old.cc
  hipStreamBeginCapture_old.cc
  hipStreamIsCapturing.cc
  hipStreamIsCapturing_old.cc
  hipGraphMemFreeNodeGetParams.cc
  hipGraphAsyncUserObj.cc
  hipGraphExecBatchMemOpNodeSetParams.cc
  hipGraphAddBatchMemOpNode.cc
  hipGraphBatchMemOpNodeSetParams.cc
  hipDrvGraphExecMemcpyNodeSetParams.cc
  hipDrvGraphAddMemFreeNode.cc)

if(HIP_PLATFORM MATCHES "amd")
  set(AMD_SRC
    hipStreamCaptureExtModuleLaunchKernel.cc
    hipStreamBeginCaptureToGraph.cc
    hipGetProcAddressGraphApis.cc
    # Below files are disabled in NVIDIA as PSDB builds are failing due to lower CUDA version.
    hipGraphExecNodeSetParams.cc
    hipGraphNodeSetParams.cc
    hipGraphExecGetFlags.cc
    # this files are disabled for NVIDIA due to lacking corresponding hipStreamGetCaptureInfo_v3 implementation
    hipStreamGetCaptureInfo_old.cc
    hipStreamGetCaptureInfo_v2.cc
    hipStreamUpdateCaptureDependencies.cc
    hipGraphAddNodeBeginCapture.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

hip_add_exe_to_target(NAME GraphsTest1
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

set(TEST_SRC
  hipStreamGetCaptureInfo.cc
  hipStreamEndCapture.cc
  hipStreamEndCapture_old.cc
  hipGraphMemcpyNodeSetParamsFromSymbol_old.cc
  hipGraphMemcpyNodeSetParamsFromSymbol.cc
  hipGraphExecEventWaitNodeSetEvent.cc
  hipGraphAddMemsetNode.cc
  hipGraphAddKernelNode.cc
  hipGraphMemcpyNodeGetParams.cc
  hipGraphMemcpyNodeGetParams_old.cc
  hipGraphMemcpyNodeSetParams.cc
  hipGraphKernelNodeGetParams.cc
  hipGraphKernelNodeSetParams.cc
  hipGraphExecKernelNodeSetParams.cc
  hipGraphLaunch.cc
  hipGraphLaunch_old.cc
  hipGraphMemcpyNodeSetParams1D.cc
  hipGraphMemcpyNodeSetParams1D_old.cc
  hipGraphExecMemcpyNodeSetParamsToSymbol_old.cc
  hipGraphExecMemcpyNodeSetParamsToSymbol.cc
  hipGraphNodeGetDependentNodes.cc
  hipGraphNodeGetDependencies.cc
  hipGraphHostNodeGetParams.cc
  hipGraphExecChildGraphNodeSetParams.cc
  hipGraphCreate.cc
  hipGraphDestroy.cc
  hipThreadExchangeStreamCaptureMode.cc
  hipLaunchHostFuncWithGraph.cc
  hipUserObjectCreate.cc
  hipUserObjectRelease.cc
  hipUserObjectRetain.cc
  hipGraphReleaseUserObject.cc
  hipGraphRetainUserObject.cc
  hipGraphDebugDotPrint.cc
  hipGraphCloneComplx.cc
  hipGraphNodeSetEnabled.cc
  hipGraphNodeGetEnabled.cc
  hipGraphExecDestroy.cc
  hipGraphUpload.cc
  hipGraphKernelNodeCopyAttributes.cc
  hipGraphCycle.cc
  hipGraphPerf.cc
  hipGraphKernelNodeGetAttribute.cc
  hipGraphKernelNodeSetAttribute.cc
  hipGraphMemAllocNodeGetParams.cc
  hipGraphAddMemAllocNode.cc
  hipGraphAddMemFreeNode.cc
  hipDrvGraphMemcpyNodeGetParams.cc
  hipDrvGraphMemcpyNodeSetParams.cc
  hipDeviceSetGraphMemAttribute.cc
  hipDeviceGetGraphMemAttribute.cc
  hipDeviceGraphMemTrim.cc
  hipDrvGraphExecMemsetNodeSetParams.cc
)

if(HIP_PLATFORM MATCHES "amd")
  set(AMD_SRC
    # hipGraphAddNode, hipGraphNodeParams, hipMemcpyNodeParams are not mapped to Nvidia
    hipGraphAddNode.cc
    # hipDrvGraphAddMemsetNode, HIP_MEMSET_NODE_PARAMS are not mapped to Nvidia
    hipDrvGraphAddMemsetNode.cc
    # hipDrvGraphAddMemcpyNode not mapped to Nvidia
    hipDrvGraphAddMemcpyNode.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _HIP_BUILD_TYPE_UPPER)
  separate_arguments(HIP_DEVICE_BUILD_FLAGS_NO_ARCH NATIVE_COMMAND
    "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_${_HIP_BUILD_TYPE_UPPER}}")
  list(FILTER HIP_DEVICE_BUILD_FLAGS_NO_ARCH EXCLUDE REGEX "--offload-arch=")
  add_custom_target(add_Kernel.code COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST}
  --cuda-device-only -x hip ${CMAKE_CURRENT_SOURCE_DIR}/add_Kernel.cpp
  -o ${CMAKE_CURRENT_BINARY_DIR}/../graph/add_Kernel.code
  -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  add_custom_target(hipMatMul COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST}
    --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/hipMatMul.cc
    -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/graph/hipMatMul.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  add_dependencies(build_tests hipMatMul)
  set_property(GLOBAL APPEND PROPERTY
               G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/hipMatMul.code)
else()
  add_custom_target(add_Kernel.code COMMAND ${CMAKE_HIP_COMPILER}
  --fatbin ${OFFLOAD_ARCH_LIST} -x cu ${CMAKE_CURRENT_SOURCE_DIR}/add_Kernel.cpp
  -o ${CMAKE_CURRENT_BINARY_DIR}/../graph/add_Kernel.code
  -I${HIP_INCLUDE_DIR}
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
endif()

set_property(GLOBAL APPEND PROPERTY
               G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/add_Kernel.code)


hip_add_exe_to_target(NAME GraphsTest2
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

add_dependencies(build_tests add_Kernel.code)
