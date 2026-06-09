cmake_minimum_required(VERSION 3.20)

project(hip_loader_spike LANGUAGES C CXX)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(HIP_LOADER_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
cmake_path(NORMAL_PATH HIP_LOADER_SOURCE_ROOT)
set(HIP_LOADER_ANNOTATIONS
  "${CMAKE_CURRENT_LIST_DIR}/tools/hip_api_manifest/api_annotations.json")
set(HIP_LOADER_GENERATOR
  "${CMAKE_CURRENT_LIST_DIR}/tools/hip_api_manifest/generate.py")
set(HIP_LOADER_GENERATOR_MODULES
  "${CMAKE_CURRENT_LIST_DIR}/tools/hip_api_manifest/api_diff.py"
  "${CMAKE_CURRENT_LIST_DIR}/tools/hip_api_manifest/clang_extract.py")
set(HIP_LOADER_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${HIP_LOADER_GENERATOR}"
  ${HIP_LOADER_GENERATOR_MODULES}
  "${HIP_LOADER_ANNOTATIONS}"
  "${HIP_LOADER_SOURCE_ROOT}/projects/clr/hipamd/include/hip/amd_detail/amd_channel_descriptor.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/clr/hipamd/src/hip_hcc.map.in"
  "${HIP_LOADER_SOURCE_ROOT}/projects/clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/clr/hipamd/include/hip/amd_detail/amd_hip_runtime_pt_api.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/clr/hipamd/include/hip/amd_detail/hip_api_trace.hpp"
  "${HIP_LOADER_SOURCE_ROOT}/projects/clr/hipamd/include/hip/amd_detail/hip_profiler_ext.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/clr/hipamd/include/hip/amd_detail/hip_runtime_prof.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/hip/VERSION"
  "${HIP_LOADER_SOURCE_ROOT}/projects/hip/include/hip/hip_api_symbols.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/hip/include/hip/hip_deprecated.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/hip/include/hip/hip_ext.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/hip/include/hip/hip_gl_interop.h"
  "${HIP_LOADER_SOURCE_ROOT}/projects/hip/include/hip/hip_runtime_api.h")

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${HIP_LOADER_GENERATOR}" generate
          --source-root "${HIP_LOADER_SOURCE_ROOT}"
          --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
          --annotations "${HIP_LOADER_ANNOTATIONS}"
  RESULT_VARIABLE HIP_LOADER_GENERATE_RESULT
)
if(NOT HIP_LOADER_GENERATE_RESULT EQUAL 0)
  message(FATAL_ERROR "HIP loader manifest generation failed")
endif()

set(HIP_LOADER_INCLUDE_DIRS
  "${HIP_LOADER_GENERATED_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${HIP_LOADER_SOURCE_ROOT}/projects/clr/hipamd/include"
  "${HIP_LOADER_SOURCE_ROOT}/projects/hip/include")

set(HIP_LOADER_COMPILE_DEFS
  HIP_LOADER_BUILDING_LIBRARY=1
  __HIP_PLATFORM_AMD__=1
  __HIP_DISABLE_CPP_FUNCTIONS__=1)

set(HIP_LOADER_COMPILE_OPTIONS)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  list(APPEND HIP_LOADER_COMPILE_OPTIONS -Wno-deprecated-declarations)
endif()

add_library(hip_loader_platform OBJECT src/hip_loader_platform.cpp)
set_target_properties(hip_loader_platform PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hip_loader_platform PRIVATE ${HIP_LOADER_INCLUDE_DIRS})
target_compile_definitions(hip_loader_platform PRIVATE ${HIP_LOADER_COMPILE_DEFS})
target_compile_options(hip_loader_platform PRIVATE ${HIP_LOADER_COMPILE_OPTIONS})

function(configure_generated_shared target source output_name output_dir)
  add_library(${target} SHARED ${source})
  target_include_directories(${target} PRIVATE ${HIP_LOADER_INCLUDE_DIRS})
  target_compile_definitions(${target} PRIVATE ${HIP_LOADER_COMPILE_DEFS})
  target_compile_options(${target} PRIVATE ${HIP_LOADER_COMPILE_OPTIONS})
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES
    OUTPUT_NAME "${output_name}"
    LIBRARY_OUTPUT_DIRECTORY "${output_dir}"
    RUNTIME_OUTPUT_DIRECTORY "${output_dir}")
endfunction()

function(add_generated_loader target public_major generated_source version_script)
  add_library(${target} SHARED
    ${generated_source}
    src/loader_v6_compat.cpp
    $<TARGET_OBJECTS:hip_loader_platform>)
  target_include_directories(${target} PRIVATE ${HIP_LOADER_INCLUDE_DIRS})
  target_compile_definitions(${target} PRIVATE ${HIP_LOADER_COMPILE_DEFS})
  target_compile_options(${target} PRIVATE ${HIP_LOADER_COMPILE_OPTIONS})
  target_compile_features(${target} PRIVATE cxx_std_17)
  if(WIN32)
    set_target_properties(${target} PROPERTIES
      OUTPUT_NAME "amdhip-${public_major}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/v${public_major}")
  else()
    set_target_properties(${target} PROPERTIES
      OUTPUT_NAME "amdhip64"
      SOVERSION "${public_major}"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/v${public_major}")
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    target_link_options(${target} PRIVATE "LINKER:--version-script=${version_script}")
    set_target_properties(${target} PROPERTIES LINK_DEPENDS "${version_script}")
  endif()
endfunction()

add_generated_loader(
  hip_loader_v6 6
  "${HIP_LOADER_GENERATED_DIR}/loader_v6.cpp"
  "${HIP_LOADER_GENERATED_DIR}/hip_loader_v6.map")
add_generated_loader(
  hip_loader_v7 7
  "${HIP_LOADER_GENERATED_DIR}/loader_v7.cpp"
  "${HIP_LOADER_GENERATED_DIR}/hip_loader_v7.map")

configure_generated_shared(
  hip_test_backend
  "${HIP_LOADER_GENERATED_DIR}/test_backend.cpp"
  "amdhip-test-backend"
  "${CMAKE_CURRENT_BINARY_DIR}/backend")
configure_generated_shared(
  hip_test_backend_wrong_major
  "${HIP_LOADER_GENERATED_DIR}/test_backend_wrong_major.cpp"
  "amdhip-test-backend-wrong-major"
  "${CMAKE_CURRENT_BINARY_DIR}/backend")
configure_generated_shared(
  hip_test_backend_missing_handshake
  "${HIP_LOADER_GENERATED_DIR}/test_backend_missing_handshake.cpp"
  "amdhip-test-backend-missing-handshake"
  "${CMAKE_CURRENT_BINARY_DIR}/backend")
configure_generated_shared(
  hip_test_backend_bad_public_export
  "${HIP_LOADER_GENERATED_DIR}/test_backend_bad_public_export.cpp"
  "amdhip-test-backend-bad-public-export"
  "${CMAKE_CURRENT_BINARY_DIR}/backend")
configure_generated_shared(
  hip_test_backend_bad_compiler_export
  "${HIP_LOADER_GENERATED_DIR}/test_backend_bad_compiler_export.cpp"
  "amdhip-test-backend-bad-compiler-export"
  "${CMAKE_CURRENT_BINARY_DIR}/backend")
configure_generated_shared(
  hip_test_preload_interposer
  "${HIP_LOADER_GENERATED_DIR}/test_preload_interposer.cpp"
  "amdhip-test-preload-interposer"
  "${CMAKE_CURRENT_BINARY_DIR}/backend")

if(BUILD_TESTING)
  add_test(NAME hip_loader_manifest_check
    COMMAND "${Python3_EXECUTABLE}" "${HIP_LOADER_GENERATOR}" check
            --source-root "${HIP_LOADER_SOURCE_ROOT}"
            --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
            --annotations "${HIP_LOADER_ANNOTATIONS}"
            --ast)

  add_test(NAME hip_loader_manifest_diff
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_manifest_diff.py")

  add_test(NAME hip_loader_ctypes
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hip_loader.py"
      --manifest "${HIP_LOADER_GENERATED_DIR}/hip_api_manifest.json"
      --v6-loader $<TARGET_FILE:hip_loader_v6>
      --v7-loader $<TARGET_FILE:hip_loader_v7>
      --backend $<TARGET_FILE:hip_test_backend>
      --wrong-major-backend $<TARGET_FILE:hip_test_backend_wrong_major>
      --missing-handshake-backend $<TARGET_FILE:hip_test_backend_missing_handshake>
      --bad-public-export-backend $<TARGET_FILE:hip_test_backend_bad_public_export>
      --bad-compiler-export-backend $<TARGET_FILE:hip_test_backend_bad_compiler_export>
      --preload-interposer $<TARGET_FILE:hip_test_preload_interposer>)
endif()
