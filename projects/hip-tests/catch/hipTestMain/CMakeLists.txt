# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
  add_definitions(-DHT_LOG_ENABLE)
endif()

set(MAIN_SRC
    main.cc
    hip_test_context.cc
    hip_test_features.cc
    hip_test_params.cc
    hip_test_listener.cc)
add_library(Main_Object EXCLUDE_FROM_ALL OBJECT ${MAIN_SRC})
set_source_files_properties(${MAIN_SRC} PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
if(ENABLE_SPIRV)
  target_compile_definitions(Main_Object PRIVATE ENABLE_SPIRV)
endif()
target_link_libraries(Main_Object PRIVATE Catch2::Catch2 hip::host hip::device)

# Include path for generated headers (hip_test_parameters.hh)
target_include_directories(Main_Object PRIVATE ${CATCH_INCLUDE_BINARY_DIR})

add_dependencies(Main_Object hip_tests_config)
