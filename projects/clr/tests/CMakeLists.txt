# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(ROCCLR_ENABLE_HSA)
  add_executable(rocclr-isa-policy-test isa_policy_test.cpp)
  target_link_libraries(rocclr-isa-policy-test PRIVATE rocclr)
  add_test(NAME rocclr-isa-policy COMMAND rocclr-isa-policy-test)
endif()

if(CLR_BUILD_HIP AND HIP_PLATFORM STREQUAL "amd")
  find_package(hsa-runtime64 1.11 REQUIRED CONFIG)
  add_executable(hip-isa-views-test isa_views_test.cpp)
  target_compile_definitions(hip-isa-views-test PRIVATE __HIP_PLATFORM_AMD__)
  target_include_directories(hip-isa-views-test PRIVATE
    ${HIP_COMMON_DIR}/include
    ${PROJECT_SOURCE_DIR}/hipamd/include
    ${PROJECT_BINARY_DIR}/hipamd/include)
  target_link_libraries(hip-isa-views-test PRIVATE
    amdhip64
    hsa-runtime64::hsa-runtime64)
  add_test(NAME hip-isa-views-native COMMAND hip-isa-views-test)
  set_tests_properties(hip-isa-views-native PROPERTIES SKIP_RETURN_CODE 77)
endif()
