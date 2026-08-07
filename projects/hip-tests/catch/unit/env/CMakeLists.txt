# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Environment Variable Tests (AMD only)
if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      hipEnvGpuEnablePal.cc
  )

  # Helper executable to check runtime path (PAL vs ROCr)
  add_executable(hipSetEnv_helper EXCLUDE_FROM_ALL hipSetEnv_helper.cc)
  set_source_files_properties(hipSetEnv_helper.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipSetEnv_helper)
  target_link_libraries(hipSetEnv_helper ${GPGPU_LINKER_LIBRARIES})

  hip_add_exe_to_target(NAME EnvironmentTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests)

  add_dependencies(EnvironmentTest hipSetEnv_helper)
endif()
