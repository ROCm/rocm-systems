# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This target deliberately compiles only the environment parser and its tests.
# It must stay independent of HIP, HSA, MPI, and the rocSHMEM runtime library so
# that a host-ASAN artifact can be validated on a CPU-only runner.
find_package(GTest CONFIG REQUIRED)

add_executable(rocshmem_envvar_test
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit_tests/envvar_gtest.cpp
  ${CMAKE_SOURCE_DIR}/src/envvar.cpp
)
target_include_directories(rocshmem_envvar_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../unit_tests
  ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(rocshmem_envvar_test PRIVATE GTest::gtest GTest::gtest_main)

target_compile_options(rocshmem_envvar_test PRIVATE
  $<$<BOOL:${ASAN}>:-fsanitize=address;-g>
)
target_link_options(rocshmem_envvar_test PRIVATE
  $<$<BOOL:${ASAN}>:-fsanitize=address;-g;-shared-libasan>
)
if(ASAN)
  set_target_properties(rocshmem_envvar_test PROPERTIES
    INSTALL_RPATH "${ASAN_RUNTIME_DIR}")
endif()

add_test(NAME rocshmem_envvar_test COMMAND rocshmem_envvar_test)
set_tests_properties(rocshmem_envvar_test PROPERTIES
  LABELS "quick;standard;pr;pre-commit"
  TIMEOUT 60)

install(TARGETS rocshmem_envvar_test
  COMPONENT tests
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
