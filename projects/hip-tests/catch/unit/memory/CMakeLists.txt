# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Common Tests - Test independent of all platforms
set(COMMON_SHARED_SRC DriverContext.cc)

# FIXME: cjatin
# This is a temp hack to pass it on windows since it has a char limit for
# command line. Eventually we need to use rsp files with ninja on windows.
# But this exposes several problems with hipcc and build system on windows.
# While those are fixed, this hack should allow us to pass windows CI.
set(TEST_SRC
    memset.cc
    malloc.cc
    hipMemcpy2DToArray.cc
    hipMemcpy2DToArray_old.cc
    hipMemcpy2DToArrayAsync.cc
    hipMemcpy2DToArrayAsync_old.cc
    hipMemcpy3D.cc
    hipMemcpy3D_old.cc
    hipMemcpy3DAsync.cc
    hipMemcpy3DAsync_old.cc
    hipMemcpyParam2D.cc
    hipMemcpyParam2D_old.cc
    hipMemcpyParam2DAsync.cc
    hipMemcpyParam2DAsync_old.cc
    hipMemcpy2D.cc
    hipMemcpy2D_old.cc
    hipMemcpy2DAsync.cc
    hipMemcpy2DAsync_old.cc
    hipMemcpy2DFromArray.cc
    hipMemcpy2DFromArray_old.cc
    hipMemcpy2DFromArrayAsync.cc
    hipMemcpy2DFromArrayAsync_old.cc
    hipMemcpyAtoD.cc
    hipMemcpyAtoH.cc
    hipMemcpyAtoHAsync.cc
    hipMemcpyDtoA.cc
    hipMemcpyAtoH_old.cc
    hipMemcpyHtoA.cc
    hipMemcpyHtoAAsync.cc
    hipMemcpyAtoA.cc
    hipMemcpyHtoA_old.cc
    hipMemcpyAllApiNegative.cc
    hipMemcpy_MultiThread.cc
    hipHostRegister.cc
    hipHostUnregister.cc
    hipHostGetFlags.cc
    hipHostGetDevicePointer.cc
    hipMallocHost.cc
    hipMemAllocHost.cc
    hipMallocManaged_MultiScenario.cc
    hipMemsetNegative.cc
    hipMemset.cc
    hipMemset3D.cc
    hipMemset2D.cc
    hipHostMallocTests.cc
    hipMemset3DFunctional.cc
    hipMemset3DRegressMultiThread.cc
    hipMallocManagedFlagsTst.cc
    hipMemPrefetchAsyncExtTsts.cc
    hipMemAdviseMmap.cc
    hipMallocManaged.cc
    hipMemRangeGetAttribute.cc
    hipMemRangeGetAttribute_old.cc
    hipMemCoherencyTst.cc
    hipMemsetD32.cc
    hipMemsetD16.cc
    hipMemsetD16Async.cc
    hipMemsetD32Async.cc
    hipMemsetD8Async.cc
    hipMemcpy2DArrayToArray.cc
    hipMemcpy3DBatchAsync.cc
    hipMemcpyBatchAsync.cc
    hipMemsetD2D8.cc
    hipMemsetD2D8Async.cc
    hipMemsetD2D16.cc
    hipMemsetD2D16Async.cc
    hipMemsetD2D32.cc
    hipMemsetD2D32Async.cc
    hipMemcpy3DPeer.cc
    hipMemcpy3DPeerAsync.cc
    )

if(UNIX)
  set(TEST_SRC ${TEST_SRC} hipMemPoolExportPointer.cc
  hipMemPoolExportToShareableHandle.cc
  hipMemPoolImportFromShareableHandle.cc
  hipMemPoolImportPointer.cc)
endif()

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      ${TEST_SRC}
      hipMemPtrGetInfo.cc
      hipPointerGetAttributes.cc
      hipExtMallocWithFlags.cc
      hipMallocMngdMultiThread.cc
      hipMemVmm.cc
      hipArray.cc
      hipMemcpyDeviceToDeviceNoCU.cc
      hipGetProcAddressMemoryApis.cc
      # Will be enabled for NVIDIA after fix for SWDEV-551244
      hipMemPrefetchAsync_v2.cc
      hipMemAdvise_v2.cc)
  if(UNIX)
    # Should be compiled for NVIDIA as well after EXSWHTEC-346 is addressed
    # For windows build error occurs undefined symbol: hipPointerSetAttribute
    set(TEST_SRC ${TEST_SRC} hipPointerSetAttribute.cc)
  endif()
else()
  set(TEST_SRC ${TEST_SRC} hipGetSymbolSizeAddress.cc)
endif()

hip_add_exe_to_target(NAME MemoryTest1
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

if(UNIX)
  # link libnuma for numa_available(), numa_max_node(), move_pages(), etc.
  find_library(NUMA_LIB numa)
  if(NUMA_LIB)
    target_link_libraries(MemoryTest1 ${NUMA_LIB})
  else()
    message(WARNING "libnuma not found; HostNuma tests will fail to link")
  endif()
endif()

if(HIP_PLATFORM MATCHES "amd")
  set_source_files_properties(hipHostRegister.cc PROPERTIES COMPILE_FLAGS -std=c++17)
  add_executable(hipHostRegisterPerf EXCLUDE_FROM_ALL hipHostRegister_exe.cc)
  add_dependencies(build_tests hipHostRegisterPerf)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipHostRegisterPerf)
if(UNIX)
  add_executable(hipMemAdviseTstAlignedAllocMem EXCLUDE_FROM_ALL hipMemAdvise_AlignedAllocMem_Exe.cc)
  add_dependencies(MemoryTest1 hipMemAdviseTstAlignedAllocMem)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipMemAdviseTstAlignedAllocMem)
endif()
endif()

set(TEST_SRC
    hipMemcpyFromSymbol.cc
    hipPtrGetAttribute.cc
    hipMemPoolApi.cc
    hipDeviceGetMemPool.cc
    hipDeviceSetMemPool.cc
    hipMemPoolSetGetAccess.cc
    hipMemPoolSetGetAttribute.cc
    hipMemPoolCreate.cc
    hipMemPoolDestroy.cc
    hipMemPoolMaxAlloc.cc
    hipMemPoolTrimTo.cc
    hipMallocFromPoolAsync.cc
    hipMemcpyPeer.cc
    hipMemcpyPeer_old.cc
    hipMemcpyPeerAsync.cc
    hipMemcpyPeerAsync_old.cc
    hipMemcpyWithStream_old.cc
    hipMemcpyWithStream.cc
    hipMemcpyWithStreamMultiThread.cc
    hipMemsetAsyncAndKernel.cc
    hipMemset2DAsyncMultiThreadAndKernel.cc
    hipMallocConcurrency.cc
    hipMemcpyDtoD.cc
    hipMemcpyDtoDAsync.cc
    hipHostMalloc.cc
    hipMemcpy_EdgeCases.cc
    hipMemcpy_derivatives.cc
    hipMemcpyAsync.cc
    hipMemcpyAsync_old.cc
    hipMemcpyAsync_derivatives.cc
    hipMemsetFunctional.cc
    hipMalloc.cc
    hipMallocPitch.cc
    hipMalloc3D.cc
    hipMalloc3DArray.cc
    hipArray3DCreate.cc
    hipDrvMemcpy3D.cc
    hipDrvMemcpy3D_old.cc
    hipDrvMemcpy3DAsync.cc
    hipDrvMemcpy3DAsync_old.cc
    hipPointerGetAttribute.cc
    hipDrvPtrGetAttributes.cc
    hipMemPrefetchAsync.cc
    hipMemGetInfo.cc
    hipFree.cc
    hipFreeArray.cc
    hipHostFree.cc
    hipMemcpySync.cc
    hipMemsetSync.cc
    hipMemsetAsync.cc
    hipMemAdvise_old.cc
    hipMemAdvise.cc
    hipMemRangeGetAttributes.cc
    hipFreeAsync.cc
    hipFreeHost.cc
    hipMallocAsync.cc
    hipStreamAttachMemAsync.cc
    hipMemRangeGetAttributes_old.cc
    hipMemGetAddressRange.cc
    hipMallocMipmappedArray.cc
    hipFreeMipmappedArray.cc
    hipHostAlloc.cc)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      ${TEST_SRC}
      # Test below currently doesn't work on NVIDIA, will update based on ticket response
      hipDrvMemcpy2DUnaligned.cc
      # Below 3 tests should be compiled for NVIDIA as well after EXSWHTEC-349 is addressed
      hipArrayGetInfo.cc
      hipArrayGetDescriptor.cc
      hipArray3DGetDescriptor.cc)
endif()

set(NOT_FOR_gfx90a_AND_ABOVE_TEST hipMallocArray.cc hipArrayCreate.cc) # tests not for gfx90a+
set(gfx90a_AND_ABOVE_TARGETS gfx90a gfx942 gfx950)
function(CheckRejectedArchs OFFLOAD_ARCH_STR_LOCAL)
  set(ARCH_CHECK -1 PARENT_SCOPE)
  set(NOT_GFX90a -1)
  set(GFX90a -1)
  string(REGEX MATCHALL "--offload-arch=gfx[0-9a-z]+" OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR_LOCAL})
  foreach(OFFLOAD_ARCH IN LISTS OFFLOAD_ARCH_LIST)
    string(REGEX MATCHALL "--offload-arch=(gfx[0-9a-z]+)" matches ${OFFLOAD_ARCH})
    if (CMAKE_MATCH_COUNT EQUAL 1)
      if (CMAKE_MATCH_1 IN_LIST gfx90a_AND_ABOVE_TARGETS)
        set(GFX90a 1)
      else()
        set(NOT_GFX90a 1)
      endif()         # CMAKE_MATCH_1
    endif()        # CMAKE_MATCH_COUNT
  endforeach()   # OFFLOAD_ARCH_LIST
  if (${NOT_GFX90a} EQUAL -1 AND ${GFX90a} EQUAL 1)
    set(ARCH_CHECK 1 PARENT_SCOPE)
  endif()
endfunction()  # CheckAcceptedArchs
if(HIP_PLATFORM MATCHES "amd")
  if (DEFINED OFFLOAD_ARCH_STR)
    CheckRejectedArchs(${OFFLOAD_ARCH_STR})
  elseif(DEFINED $ENV{HCC_AMDGPU_TARGET})
    CheckRejectedArchs($ENV{HCC_AMDGPU_TARGET})
  else()
    set(ARCH_CHECK -1)
  endif()
  if(${ARCH_CHECK} EQUAL -1)
    message(STATUS "Adding test: ${NOT_FOR_gfx90a_AND_ABOVE_TEST}")
    set(TEST_SRC ${TEST_SRC} ${NOT_FOR_gfx90a_AND_ABOVE_TEST})
  endif()
else()
  set(TEST_SRC ${TEST_SRC} ${NOT_FOR_gfx90a_AND_ABOVE_TEST})
endif()

hip_add_exe_to_target(NAME MemoryTest2
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
  inlineVar.cc
  memoryCommon.cc
)

hip_add_exe_to_target(NAME InlineVarTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

  set_target_properties(InlineVarTest PROPERTIES COMPILE_FLAGS -fgpu-rdc)
  set_target_properties(InlineVarTest PROPERTIES LINK_FLAGS -fgpu-rdc)
endif()

set(TEST_SRC
  hipSVMTestByteGranularity.cpp
  hipSVMTestFineGrainMemoryConsistency.cpp
  hipSVMTestFineGrainSyncBuffers.cpp
  hipSVMTestSharedAddressSpaceFineGrain.cpp
)

hip_add_exe_to_target(NAME SVMAtomicTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

if(HIP_PLATFORM MATCHES "nvidia")
  set_target_properties(SVMAtomicTest PROPERTIES COMPILE_FLAGS -arch=sm_70)
  set_target_properties(MemoryTest1 PROPERTIES COMPILE_FLAGS -arch=sm_70)
  set_target_properties(MemoryTest2 PROPERTIES COMPILE_FLAGS -arch=sm_70)
endif()
