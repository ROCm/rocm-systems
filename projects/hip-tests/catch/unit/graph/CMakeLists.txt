# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

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
  hipGraphMemsetNodeFlatAlloc.cc
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
  hipDrvGraphAddMemFreeNode.cc
  hipGraphAddNodeConcurrent.cc)

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
  hipGraphAllocNodeMemReuse.cc
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
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/add_Kernel.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST}
    --cuda-device-only -x hip ${CMAKE_CURRENT_SOURCE_DIR}/add_Kernel.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/add_Kernel.code
    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/add_Kernel.cpp
    COMMENT "Compiling add_Kernel.code")
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hipMatMul.code
    COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST}
    --cuda-device-only
    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/hipMatMul.cc
    -o ${CMAKE_CURRENT_BINARY_DIR}/hipMatMul.code
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
    -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hipMatMul.cc
    COMMENT "Compiling hipMatMul.code")
  add_custom_target(hipMatMul DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hipMatMul.code)
  set_property(GLOBAL APPEND PROPERTY
               G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/hipMatMul.code)
  if(UNIX)
    set(TEST_KERNEL_ARG_PREFETCH_SRC hipSimpleGraphWithKernel.cc)
    hip_add_exe_to_target(NAME GraphsKernelArgPrefetchTest
                          TEST_SRC ${TEST_KERNEL_ARG_PREFETCH_SRC}
                          TEST_TARGET_NAME build_tests)
    set_target_properties(GraphsKernelArgPrefetchTest PROPERTIES COMPILE_FLAGS "-DKERNEL_ARG_PREFETCH -mllvm -amdgpu-kernarg-preload-count=16")
  endif()
else()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/add_Kernel.code
    COMMAND ${CMAKE_HIP_COMPILER}
    --fatbin ${OFFLOAD_ARCH_LIST} -x cu ${CMAKE_CURRENT_SOURCE_DIR}/add_Kernel.cpp
    -o ${CMAKE_CURRENT_BINARY_DIR}/add_Kernel.code
    -I${HIP_INCLUDE_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/add_Kernel.cpp
    COMMENT "Compiling add_Kernel.code")
endif()
add_custom_target(add_Kernel DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/add_Kernel.code)

set_property(GLOBAL APPEND PROPERTY
               G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/add_Kernel.code)


hip_add_exe_to_target(NAME GraphsTest2
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

# Wire kernel artifacts to the consuming test exes so `make GraphsTest1` /
# `make GraphsTest2` builds all their dependencies. The kernels are loaded at
# runtime by tests in either binary, so wire to both.
add_dependencies(GraphsTest1 add_Kernel)
add_dependencies(GraphsTest2 add_Kernel)
if(HIP_PLATFORM MATCHES "amd")
  add_dependencies(GraphsTest1 hipMatMul)
  add_dependencies(GraphsTest2 hipMatMul)
  if(UNIX)
    add_dependencies(GraphsKernelArgPrefetchTest add_Kernel)
  endif()
endif()
