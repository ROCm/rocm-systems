################################################################################
# hip-remote-client
#
# Remote HIP runtime client library.
# Supported platforms: Windows, macOS, Linux (without native AMD GPU)
# This library implements the HIP API by forwarding calls to a remote
# HIP worker service running on a Linux system with AMD GPUs.
################################################################################

cmake_minimum_required(VERSION 3.25)
project(hip-remote-client VERSION 1.0.0 LANGUAGES C)

# Only build on macOS/Windows (or when explicitly requested for testing)
if(NOT APPLE AND NOT WIN32 AND NOT HIP_REMOTE_FORCE_BUILD)
  message(STATUS "hip-remote-client: Skipping (not macOS/Windows)")
  return()
endif()

message(STATUS "hip-remote-client: Building remote HIP client library")

################################################################################
# ROCm SDK headers (for hipDeviceProp_t and other HIP types)
#
# Pass -DROCM_PATH=$(rocm-sdk path --root) to point to the SDK root.
################################################################################

set(ROCM_PATH "" CACHE PATH "ROCm SDK root directory containing include/hip/")

if(NOT ROCM_PATH OR NOT EXISTS "${ROCM_PATH}/include/hip/hip_runtime_api.h")
  message(FATAL_ERROR
    "ROCM_PATH not set or HIP headers not found.\n"
    "Set -DROCM_PATH=<path> to your ROCm SDK root.\n"
    "  From wheels: -DROCM_PATH=$(rocm-sdk path --root)\n"
    "  System:      -DROCM_PATH=/opt/rocm")
endif()

message(STATUS "hip-remote-client: Using HIP headers from ${ROCM_PATH}/include")

################################################################################
# Library target
################################################################################

set(HIP_REMOTE_SOURCES
  src/hip_client.c
  src/hip_api_device.c
  src/hip_api_memory.c
  src/hip_api_stream.c
  src/hip_api_module.c
  src/hip_api_graph.c
  src/hip_api_fatbin.c
  src/hip_api_stubs_gen.c
  src/hip_remote_shm.c
)

set(HIP_REMOTE_HEADERS
  include/hip_remote/hip_remote_protocol.h
  include/hip_remote/hip_remote_client.h
  include/hip_remote/hip_remote_internal.h
  include/hip_remote/hip_remote_platform.h
)

add_library(amdhip64 SHARED ${HIP_REMOTE_SOURCES})

target_include_directories(amdhip64
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(amdhip64 PRIVATE "${ROCM_PATH}/include")
target_compile_definitions(amdhip64 PRIVATE __HIP_PLATFORM_AMD__=1)

# Proxy mode: output DLL as amdhip64_7 to replace the real HIP runtime DLL
option(HIP_REMOTE_PROXY_MODE "Build as amdhip64_7 drop-in replacement" OFF)

# Set library properties
set_target_properties(amdhip64 PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

if(WIN32 AND HIP_REMOTE_PROXY_MODE)
  set_target_properties(amdhip64 PROPERTIES OUTPUT_NAME "amdhip64_7" PREFIX "")
endif()

# Platform-specific settings
if(APPLE)
  set_target_properties(amdhip64 PROPERTIES
    INSTALL_NAME_DIR "@rpath"
    BUILD_WITH_INSTALL_RPATH ON
    MACOSX_RPATH ON
  )
  target_link_libraries(amdhip64 PRIVATE pthread)
elseif(WIN32)
  target_link_libraries(amdhip64 PRIVATE ws2_32)
  set_target_properties(amdhip64 PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )
else()
  target_link_libraries(amdhip64 PRIVATE pthread rt)
endif()

################################################################################
# Install
################################################################################

include(GNUInstallDirs)

install(TARGETS amdhip64
  EXPORT hip-remote-client-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/hip_remote
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create hip-config.cmake for find_package support
include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/hip-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/hip-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/hip-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/hip-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip
)

install(EXPORT hip-remote-client-targets
  FILE hip-targets.cmake
  NAMESPACE hip::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hip
)

################################################################################
# Tests
################################################################################

if(BUILD_TESTING)
  enable_testing()

  # Basic connectivity test (requires TF_WORKER_HOST to be set)
  add_executable(hip_remote_test_basic tests/test_basic.c)
  target_link_libraries(hip_remote_test_basic PRIVATE amdhip64)
  target_include_directories(hip_remote_test_basic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_basic
    COMMAND hip_remote_test_basic
  )

  # Mark as integration test (needs remote worker)
  set_tests_properties(hip_remote_basic PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Extended tests for new APIs (hipMallocAsync, hipMemcpy2D, etc.)
  add_executable(hip_remote_test_extended tests/test_extended.c)
  target_link_libraries(hip_remote_test_extended PRIVATE amdhip64)
  target_include_directories(hip_remote_test_extended PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_extended
    COMMAND hip_remote_test_extended
  )

  set_tests_properties(hip_remote_extended PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Phase 2 tests (device limits, peer access, etc.)
  add_executable(hip_remote_test_phase2 tests/test_phase2.c)
  target_link_libraries(hip_remote_test_phase2 PRIVATE amdhip64)
  target_include_directories(hip_remote_test_phase2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_phase2
    COMMAND hip_remote_test_phase2
  )

  set_tests_properties(hip_remote_phase2 PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Graph API tests
  add_executable(hip_remote_test_graphs tests/test_graphs.c)
  target_link_libraries(hip_remote_test_graphs PRIVATE amdhip64)
  target_include_directories(hip_remote_test_graphs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_graphs
    COMMAND hip_remote_test_graphs
  )

  set_tests_properties(hip_remote_graphs PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # IPC (Inter-Process Communication) tests
  add_executable(hip_remote_test_ipc tests/test_ipc.c)
  target_link_libraries(hip_remote_test_ipc PRIVATE amdhip64)
  target_include_directories(hip_remote_test_ipc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_ipc
    COMMAND hip_remote_test_ipc
  )

  set_tests_properties(hip_remote_ipc PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Memory Pool tests
  add_executable(hip_remote_test_mempool tests/test_mempool.c)
  target_link_libraries(hip_remote_test_mempool PRIVATE amdhip64)
  target_include_directories(hip_remote_test_mempool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_mempool
    COMMAND hip_remote_test_mempool
  )

  set_tests_properties(hip_remote_mempool PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Graph Node tests
  add_executable(hip_remote_test_graph_nodes tests/test_graph_nodes.c)
  target_link_libraries(hip_remote_test_graph_nodes PRIVATE amdhip64)
  target_include_directories(hip_remote_test_graph_nodes PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_graph_nodes
    COMMAND hip_remote_test_graph_nodes
  )

  set_tests_properties(hip_remote_graph_nodes PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Device APIs tests
  add_executable(hip_remote_test_device_apis tests/test_device_apis.c)
  target_link_libraries(hip_remote_test_device_apis PRIVATE amdhip64)
  target_include_directories(hip_remote_test_device_apis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_device_apis
    COMMAND hip_remote_test_device_apis
  )

  set_tests_properties(hip_remote_device_apis PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Memory APIs tests
  add_executable(hip_remote_test_memory_apis tests/test_memory_apis.c)
  target_link_libraries(hip_remote_test_memory_apis PRIVATE amdhip64)
  target_include_directories(hip_remote_test_memory_apis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_memory_apis
    COMMAND hip_remote_test_memory_apis
  )

  set_tests_properties(hip_remote_memory_apis PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Quick Win APIs tests
  add_executable(hip_remote_test_quick_wins tests/test_quick_wins.c)
  target_link_libraries(hip_remote_test_quick_wins PRIVATE amdhip64)
  target_include_directories(hip_remote_test_quick_wins PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_quick_wins
    COMMAND hip_remote_test_quick_wins
  )

  set_tests_properties(hip_remote_quick_wins PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Function Attributes tests
  add_executable(hip_remote_test_func_attrs tests/test_func_attrs.c)
  target_link_libraries(hip_remote_test_func_attrs PRIVATE amdhip64)
  target_include_directories(hip_remote_test_func_attrs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_func_attrs
    COMMAND hip_remote_test_func_attrs
  )

  set_tests_properties(hip_remote_func_attrs PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )

  # Advanced Graph APIs tests
  add_executable(hip_remote_test_graph_advanced tests/test_graph_advanced.c)
  target_link_libraries(hip_remote_test_graph_advanced PRIVATE amdhip64)
  target_include_directories(hip_remote_test_graph_advanced PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  add_test(NAME hip_remote_graph_advanced
    COMMAND hip_remote_test_graph_advanced
  )

  set_tests_properties(hip_remote_graph_advanced PROPERTIES
    LABELS "integration"
    ENVIRONMENT "TF_DEBUG=1"
  )
endif()
