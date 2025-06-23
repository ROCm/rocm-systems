# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

# Common Tests - Test independent of all platforms
set(TEST_SRC
    floatMath.cc
    anyAll.cc
    ballot.cc
    clz.cc
    ffs.cc
    funnelshift.cc
    brev.cc
    popc.cc
    ldg.cc
    threadfence_system.cc
    syncthreadsand.cc
    syncthreadscount.cc
    syncthreadsor.cc
    deviceAllocation.cc
    Atomic_func.cc
    DoublePrecisionIntrinsics.cc
    DoublePrecisionMathDevice.cc
    DoublePrecisionMathHost.cc
    FloatMathPrecise.cc
    IntegerIntrinsics.cc
    SinglePrecisionIntrinsics.cc
    SinglePrecisionMathDevice.cc
    SinglePrecisionMathHost.cc
    SimpleAtomicsTest.cc
    hipTestAtomicAdd.cc
    hipStdComplex.cc
    hipTestClock.cc
    hip_trig.cc
    hipDeviceMemcpy.cc
    hipTestIncludeMath.cc
    hipTestDotFunctions.cc
    hipTestDeviceSymbol.cc
    hipTestNew.cc
    hipThreadFence.cc
    hipTestDevice.cc
    hipTestDeviceLimit.cc
    hipTestDeviceDouble.cc
    hipTestHost.cc
    hadd.cc
)
if(HIP_PLATFORM MATCHES "nvidia")
  set_source_files_properties(hipTestHost.cc PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
endif()

# AMD only tests
set(AMD_TEST_SRC
    unsafeAtomicAdd.cc
    mbcnt.cc
    bitExtract.cc
    bitInsert.cc
    floatTM.cc
    hipMathFunctions.cc
    hmax_hmin.cc
    hipBfloat16.cc
    hipVectorTypesHost.cc
    hipVectorTypesDevice.cc
    hipTestHalf.cc
    hipComplex.cc
    hipTestFMA.cc
    hipTestNativeHalf.cc
    hip_test_make_type.cc
    bfloat16.cc
    AtomicsWithRandomActiveLanesInWavefront.cc
    fp16_ops.cc
    fp8_host.cc
    fp6_ocp.cc
    fp4_ocp.cc
)

set(AMD_ARCH_SPEC_TEST_SRC
    AtomicAdd_Coherent_withunsafeflag.cc
    AtomicAdd_Coherent_withoutflag.cc
    AtomicAdd_Coherent_withnoUnsafeflag.cc
    AtomicAdd_NonCoherent_withoutflag.cc
    AtomicAdd_NonCoherent_withnoUnsafeflag.cc
    AtomicAdd_NonCoherent_withunsafeflag.cc
    BuiltIns_fmax.cc
    BuiltIns_fmin.cc
    BuiltIns_fadd.cc
    unsafeAtomicAdd_RTC.cc
    unsafeAtomicAdd_Coherent_withunsafeflag.cc
    unsafeAtomicAdd_Coherent_withoutflag.cc
    unsafeAtomicAdd_Coherent_withnounsafeflag.cc
    unsafeAtomicAdd_NonCoherent_withoutflag.cc
    unsafeAtomicAdd_NonCoherent_withnounsafeflag.cc
    unsafeAtomicAdd_NonCoherent_withunsafeflag.cc
)

set(AMD_GFX940_SPEC_TEST_SRC
    fp8_fnuz.cc
)

set(AMD_GFX1200_SPEC_TEST_SRC
    fp8_ocp.cc
)

# Note to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx900  --offload-arch=gfx906"
# having space at the start/end of OFFLOAD_ARCH_STR can cause build failures
add_custom_target(kerDevAllocMultCO.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/kerDevAllocMultCO.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/deviceLib/kerDevAllocMultCO.code
                  -I${HIP_PATH}/include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --hip-path=${HIP_PATH})
add_custom_target(kerDevWriteMultCO.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/kerDevWriteMultCO.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/deviceLib/kerDevWriteMultCO.code
                  -I${HIP_PATH}/include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --hip-path=${HIP_PATH})
add_custom_target(kerDevFreeMultCO.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/kerDevFreeMultCO.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/deviceLib/kerDevFreeMultCO.code
                  -I${HIP_PATH}/include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --hip-path=${HIP_PATH})
add_custom_target(kerDevAllocSingleKer.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/kerDevAllocSingleKer.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/deviceLib/kerDevAllocSingleKer.code
                  -I${HIP_PATH}/include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --hip-path=${HIP_PATH})

set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/kerDevAllocSingleKer.code)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/kerDevFreeMultCO.code)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/kerDevWriteMultCO.code)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/kerDevAllocMultCO.code)

# Accepted archs to compile this cmake file
set(ACCEPTED_OFFLOAD_ARCHS gfx90a gfx942)
set(ACCEPTED_GFX940_ARCH gfx942)
set(ACCEPTED_GFX1200_ARCH gfx1200 gfx1201)
function(CheckAcceptedArchs OFFLOAD_ARCH_STR_LOCAL)
  set(ARCH_CHECK -1 PARENT_SCOPE)
  set(ARCH_GFX940 -1 PARENT_SCOPE)
  set(ARCH_GFX1200 -1 PARENT_SCOPE)
  string(REGEX MATCHALL "--offload-arch=gfx[0-9a-z]+" OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR_LOCAL})
  foreach(OFFLOAD_ARCH IN LISTS OFFLOAD_ARCH_LIST)
    string(REGEX MATCHALL "--offload-arch=(gfx[0-9a-z]+)" matches ${OFFLOAD_ARCH})
    if (CMAKE_MATCH_COUNT EQUAL 1)
      if (CMAKE_MATCH_1 IN_LIST ACCEPTED_OFFLOAD_ARCHS)
        set(ARCH_CHECK 1 PARENT_SCOPE)
      endif()         # CMAKE_MATCH_1
      if (CMAKE_MATCH_1 IN_LIST ACCEPTED_GFX940_ARCH)
        set(ARCH_GFX940 1 PARENT_SCOPE)
      endif()
      if (CMAKE_MATCH_1 IN_LIST ACCEPTED_GFX1200_ARCH)
        set(ARCH_GFX1200 1 PARENT_SCOPE)
      endif()
    endif()        # CMAKE_MATCH_COUNT
  endforeach()   # OFFLOAD_ARCH_LIST
endfunction()  # CheckAcceptedArchs

if(HIP_PLATFORM MATCHES "amd")
    if (DEFINED OFFLOAD_ARCH_STR)
      CheckAcceptedArchs(${OFFLOAD_ARCH_STR})
    elseif(DEFINED $ENV{HCC_AMDGPU_TARGET})
      CheckAcceptedArchs($ENV{HCC_AMDGPU_TARGET})
    else()
      set(ARCH_CHECK -1)
      set(ARCH_GFX940 -1)
      set(ARCH_GFX1200 -1)
    endif()
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
    set_source_files_properties(floatTM.cc PROPERTIES COMPILE_FLAGS -std=c++17)
    set_source_files_properties(bfloat16.cc PROPERTIES COMPILE_FLAGS "-DHIP_ENABLE_WARP_SYNC_BUILTINS")
if(${ARCH_CHECK} GREATER_EQUAL 0)
	set(TEST_SRC ${TEST_SRC} ${AMD_ARCH_SPEC_TEST_SRC})
    set_source_files_properties(AtomicAdd_Coherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_NonCoherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_Coherent_withnoUnsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_NonCoherent_withnoUnsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(unsafeAtomicAdd_Coherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(unsafeAtomicAdd_NonCoherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(unsafeAtomicAdd_Coherent_withnounsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(unsafeAtomicAdd_NonCoherent_withnounsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(hipMathFunctions.cc PROPERTIES COMPILE_FLAGS "-Xclang -fallow-half-arguments-and-returns")
    file(GLOB AtomicAdd_files *AtomicAdd_*_*.cc)
    set_property(SOURCE ${AtomicAdd_files} PROPERTY COMPILE_FLAGS --save-temps)
    file(GLOB unsafeAtomicAdd_files *unsafeAtomicAdd_*_*.cc)
    set_property(SOURCE ${unsafeAtomicAdd_files} PROPERTY COMPILE_FLAGS --save-temps)
endif()
if(${ARCH_GFX940} GREATER_EQUAL 0)
  set(TEST_SRC ${TEST_SRC} ${AMD_GFX940_SPEC_TEST_SRC})
endif()
if(${ARCH_GFX1200} GREATER_EQUAL 0)
  set(TEST_SRC ${TEST_SRC} ${AMD_GFX1200_SPEC_TEST_SRC})
endif()
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
elseif(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS --Wno-deprecated-declarations)
endif()

add_dependencies(build_tests kerDevAllocMultCO.code kerDevWriteMultCO.code kerDevFreeMultCO.code kerDevAllocSingleKer.code)
