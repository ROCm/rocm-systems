# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipOccupancyMaxActiveBlocksPerMultiprocessor.cc
  hipOccupancyMaxActiveBlocksPerMultiprocessor_old.cc
  hipOccupancyMaxPotentialBlockSize.cc
  hipOccupancyMaxPotentialBlockSize_old.cc
  hipModuleOccupancyMaxPotentialBlockSize.cc
  hipModuleOccupancyMaxPotentialBlockSizeWithFlags.cc
  hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.cc
  hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.cc
  hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags.cc
  hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.cc
)

hip_add_device_code(
  TARGET simple_kernel
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/simple_kernel.cc)

hip_add_exe_to_target(NAME OccupancyTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

add_dependencies(OccupancyTest simple_kernel)
