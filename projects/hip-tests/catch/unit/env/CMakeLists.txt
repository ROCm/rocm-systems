# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Environment Variable Tests (AMD only)
if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      hipEnvGpuEnablePal.cc
  )

  # Helper executable to check runtime path (PAL vs ROCr)
  add_executable(hipEnvGpuEnablePal_CheckRuntime EXCLUDE_FROM_ALL hipEnvGpuEnablePal_CheckRuntime.cc)
  set_source_files_properties(hipEnvGpuEnablePal_CheckRuntime.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipEnvGpuEnablePal_CheckRuntime)
  target_link_libraries(hipEnvGpuEnablePal_CheckRuntime ${GPGPU_LINKER_LIBRARIES})

  hip_add_exe_to_target(NAME EnvironmentTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests)

  add_dependencies(EnvironmentTest hipEnvGpuEnablePal_CheckRuntime)
endif()
