# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Environment Variable Tests
set(TEST_SRC
    hipEnvGpuEnablePal.cc
)

# Helper executable to check runtime path (PAL vs ROCr)
add_executable(hipEnvGpuEnablePal_CheckRuntime EXCLUDE_FROM_ALL hipEnvGpuEnablePal_CheckRuntime.cc)
set_source_files_properties(hipEnvGpuEnablePal_CheckRuntime.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
target_compile_options(hipEnvGpuEnablePal_CheckRuntime PRIVATE ${HIP_COMPILE_OPTIONS})
target_include_directories(hipEnvGpuEnablePal_CheckRuntime PRIVATE ${HIP_INCLUDE_DIR})
target_link_libraries(hipEnvGpuEnablePal_CheckRuntime PRIVATE ${HIP_LINK_OPTIONS})
add_dependencies(build_tests hipEnvGpuEnablePal_CheckRuntime)

hip_add_exe_to_target(NAME EnvironmentTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
