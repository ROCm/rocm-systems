# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(rj_default_compiler)

project(rocm_jit_suite)

# Default install prefix to /opt/rocm, matching other ROCm projects.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(DEFINED ENV{ROCM_PATH})
    set(CMAKE_INSTALL_PREFIX "$ENV{ROCM_PATH}" CACHE PATH "" FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "" FORCE)
  endif()
endif()

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

# HSA/KFD headers (local copies from rocr-runtime).
set(HSA_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/lib/rocjitsu/external_headers/hsa_headers")

option(RJ_BUILD_GUI "Build the simulation GUI (requires OpenGL)" ON)

include(rj_log)

# Sanitizer and static analysis options.
option(RJ_SANITIZER "Enable sanitizer (asan, ubsan, tsan, or msan)" "")
option(RJ_CLANG_TIDY "Enable clang-tidy static analysis" OFF)

if(RJ_SANITIZER)
  message(STATUS "Sanitizer enabled: ${RJ_SANITIZER}")
  if(MSVC)
    # MSVC only supports AddressSanitizer; ubsan/tsan/msan are GCC/Clang-only.
    if(RJ_SANITIZER STREQUAL "asan")
      add_compile_options(/fsanitize=address)
      # /INCREMENTAL:NO is required when using MSVC ASAN.
      add_link_options(/INCREMENTAL:NO)
    else()
      message(FATAL_ERROR
        "RJ_SANITIZER=${RJ_SANITIZER} is not supported by MSVC. Only 'asan' is available.")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    if(RJ_SANITIZER STREQUAL "asan")
      add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
      add_link_options(-fsanitize=address)
      # GCC 13+ emits false-positive -Wmaybe-uninitialized in std_function.h
      # when ASAN is enabled. Suppress to avoid -Werror failures.
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-Wno-maybe-uninitialized)
      endif()
    elseif(RJ_SANITIZER STREQUAL "ubsan")
      add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
      add_link_options(-fsanitize=undefined)
    elseif(RJ_SANITIZER STREQUAL "tsan")
      add_compile_options(-fsanitize=thread)
      add_link_options(-fsanitize=thread)
    elseif(RJ_SANITIZER STREQUAL "msan")
      add_compile_options(-fsanitize=memory -fno-omit-frame-pointer)
      add_link_options(-fsanitize=memory)
    else()
      message(FATAL_ERROR "Unknown sanitizer: ${RJ_SANITIZER}. Use asan, ubsan, tsan, or msan.")
    endif()
  else()
    message(FATAL_ERROR
      "RJ_SANITIZER is not supported for compiler '${CMAKE_CXX_COMPILER_ID}'.")
  endif()
endif()

if(RJ_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  else()
    message(WARNING "clang-tidy not found, RJ_CLANG_TIDY ignored")
  endif()
endif()

# Third-party dependencies (fetched into third_party/).
# Third-party dependencies (fetched into third_party/).
# Third-party dependencies (fetched into third_party/ by default).
# Override with -DFETCHCONTENT_BASE_DIR=/path for container builds.
if(NOT FETCHCONTENT_BASE_DIR)
  set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/third_party)
endif()
include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# FlatBuffers - serialization for config, checkpoint, and wire format.
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v24.3.25
)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest flatbuffers)

# Compile FlatBuffers schemas into C++ headers.
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(SCHEMA_FILES
    ${CMAKE_SOURCE_DIR}/schemas/simulation_config.fbs
    ${CMAKE_SOURCE_DIR}/schemas/checkpoint.fbs
)

set(GENERATED_HEADERS)
foreach(schema ${SCHEMA_FILES})
    get_filename_component(schema_name ${schema} NAME_WE)
    set(output_header "${GENERATED_DIR}/${schema_name}_generated.h")
    add_custom_command(
        OUTPUT ${output_header}
        COMMAND $<TARGET_FILE:flatc> --cpp -I ${CMAKE_SOURCE_DIR}/schemas -o ${GENERATED_DIR} ${schema}
        DEPENDS flatc ${schema}
        COMMENT "Compiling FlatBuffers schema: ${schema_name}.fbs"
    )
    list(APPEND GENERATED_HEADERS ${output_header})
endforeach()

add_custom_target(flatbuffers_schemas DEPENDS ${GENERATED_HEADERS})

# GUI dependencies (conditional).
if(RJ_BUILD_GUI)
    find_package(OpenGL QUIET)
    if(NOT OpenGL_FOUND)
        message(WARNING "OpenGL not found - disabling GUI build")
        set(RJ_BUILD_GUI OFF)
    endif()
endif()

if(RJ_BUILD_GUI)
    # GLFW - cross-platform windowing.
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)

    # Dear ImGui - immediate-mode GUI.
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.8
    )
    FetchContent_MakeAvailable(imgui)

    add_library(imgui_lib STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui_lib PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_compile_definitions(imgui_lib PUBLIC IMGUI_DEFINE_MATH_OPERATORS)
    target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)

    # imnodes - node editor for topology visualization.
    FetchContent_Declare(
        imnodes
        GIT_REPOSITORY https://github.com/Nelarius/imnodes.git
        GIT_TAG v0.5
    )
    FetchContent_MakeAvailable(imnodes)

    add_library(imnodes_lib STATIC ${imnodes_SOURCE_DIR}/imnodes.cpp)
    target_include_directories(imnodes_lib PUBLIC ${imnodes_SOURCE_DIR})
    target_link_libraries(imnodes_lib PUBLIC imgui_lib)

    # ImPlot - plotting for timeline visualization.
    FetchContent_Declare(
        implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v0.16
    )
    FetchContent_MakeAvailable(implot)

    add_library(implot_lib STATIC
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
    )
    target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
    target_link_libraries(implot_lib PUBLIC imgui_lib)
endif()

# Libraries
add_subdirectory(lib/util)
add_subdirectory(lib/simdojo)
add_subdirectory(lib/rocjitsu)

# Main shared library - assembles all object libraries.
add_library(rocjitsu_shared SHARED)
set_target_properties(rocjitsu_shared PROPERTIES OUTPUT_NAME rocjitsu)
target_link_libraries(rocjitsu_shared PRIVATE
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_cdna1 rocjitsu_isa_cdna2 rocjitsu_isa_cdna3 rocjitsu_isa_cdna4
    rocjitsu_isa_rdna1 rocjitsu_isa_rdna2 rocjitsu_isa_rdna3 rocjitsu_isa_rdna3_5 rocjitsu_isa_rdna4
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_vm_risc_v
    rocjitsu_config
    flatbuffers
    Threads::Threads
)
target_include_directories(rocjitsu_shared PRIVATE ${GENERATED_DIR})
add_dependencies(rocjitsu_shared flatbuffers_schemas)

# Tools
if(RJ_BUILD_GUI)
    add_subdirectory(tools/simgui)
endif()

# Tests and scaling tools.
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install rules.
# ---------------------------------------------------------------------------

include(GNUInstallDirs)

# Public headers: include/rocjitsu/
install(DIRECTORY lib/rocjitsu/include/rocjitsu
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

# Libraries: lib/
install(TARGETS rocjitsu_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  install(TARGETS rocjitsu_kmd_shim
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Config files and schemas: share/rocjitsu/
install(DIRECTORY configs/ DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/configs)
install(DIRECTORY schemas/ DESTINATION ${CMAKE_INSTALL_DATADIR}/rocjitsu/schemas
        FILES_MATCHING PATTERN "*.fbs")
