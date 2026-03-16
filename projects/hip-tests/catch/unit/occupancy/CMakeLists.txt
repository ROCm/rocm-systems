# Common Tests - Test independent of all platforms
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

if (HIP_PLATFORM MATCHES "amd")
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
