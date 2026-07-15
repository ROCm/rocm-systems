# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms
set(TARGETS_SUPPORTING_CLUSTERS gfx1250 gfx1251)
set(CLUSTERS_SUPPORTED -1)

function(CheckRejectedArchs OFFLOAD_ARCH_STR_LOCAL)
  string(REGEX MATCHALL "--offload-arch=gfx[0-9a-z]+" OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR_LOCAL})
  foreach(OFFLOAD_ARCH IN LISTS OFFLOAD_ARCH_LIST)
	string(REGEX MATCHALL "--offload-arch=(gfx[0-9a-z]+)" matches ${OFFLOAD_ARCH})
	if (CMAKE_MATCH_COUNT EQUAL 1)
	  if (CMAKE_MATCH_1 IN_LIST TARGETS_SUPPORTING_CLUSTERS)
		set(CLUSTERS_SUPPORTED 1 PARENT_SCOPE)
	  endif()         # CMAKE_MATCH_1
	endif()        # CMAKE_MATCH_COUNT
  endforeach()   # OFFLOAD_ARCH_LIST
endfunction()  # CheckAcceptedArchs

set(TEST_SRC
  hipOccupancyMaxActiveBlocksPerMultiprocessor.cc
  hipOccupancyMaxPotentialBlockSize.cc
  hipModuleOccupancyMaxPotentialBlockSize.cc
  hipModuleOccupancyMaxPotentialBlockSizeWithFlags.cc
  hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.cc
  hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.cc
  hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags.cc
  hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.cc
  hipOccupancyAvailableDynamicSMemPerBlock.cc
)

if(HIP_PLATFORM MATCHES "amd")
  if (DEFINED OFFLOAD_ARCH_STR)
    CheckRejectedArchs(${OFFLOAD_ARCH_STR})
  elseif(DEFINED $ENV{HCC_AMDGPU_TARGET})
    CheckRejectedArchs($ENV{HCC_AMDGPU_TARGET})
  endif()

  if(CLUSTERS_SUPPORTED EQUAL 1)
    message(STATUS "Adding clusters tests")
    set(TEST_SRC
      ${TEST_SRC}
      hipOccupancyMaxPotentialClusterSize.cc
      hipOccupancyMaxActiveClusters.cc
      )
  else()
    message(STATUS "No cluster support; clusters tests were not added")
  endif()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/simple_kernel.code
                   COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH} ${OFFLOAD_ARCH_LIST} --cuda-device-only
                     -x hip ${CMAKE_CURRENT_SOURCE_DIR}/simple_kernel.cc
                     -o simple_kernel.code
                     -I${HIP_INCLUDE_DIR} ${HIP_PATH_OPT}
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/simple_kernel.cc)
else()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/simple_kernel.code
                     COMMAND ${CMAKE_HIP_COMPILER} --fatbin ${OFFLOAD_ARCH_LIST}
                     -x cu ${CMAKE_CURRENT_SOURCE_DIR}/simple_kernel.cc
                     -o simple_kernel.code
                     -I${HIP_INCLUDE_DIR}
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/simple_kernel.cc)
endif()
add_custom_target(simple_kernel ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/simple_kernel.code)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/simple_kernel.code)
hip_add_exe_to_target(NAME OccupancyTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

add_dependencies(OccupancyTest simple_kernel)
