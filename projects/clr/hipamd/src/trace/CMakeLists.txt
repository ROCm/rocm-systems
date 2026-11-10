# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# rocm_hip API events (HIP_ENABLE_ROCM_TRACE): the generated TraceLogging probes and their
# installation, compiled into amdhip64. See README.md.

# The ROCm trace contract headers (shared/rocm-trace), in this repository.
set(CLR_ROCM_TRACE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../shared/rocm-trace"
    CACHE PATH "Path to the shared rocm-trace component")
if(NOT TARGET rocm::trace)
  if(NOT EXISTS "${CLR_ROCM_TRACE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "rocm-trace not found at ${CLR_ROCM_TRACE_DIR}. Set CLR_ROCM_TRACE_DIR "
                        "when building CLR outside the rocm-systems tree.")
  endif()
  add_subdirectory(${CLR_ROCM_TRACE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/rocm-trace)
endif()

# TraceLogging: in the Windows SDK; on Linux, the LTTng flavour of microsoft/tracelogging, whose
# CMake package brings liblttng-ust. A tracelogging::tracelogging target already defined in this
# build (one built in-tree) is used as is; otherwise the installed package is found.
if(WIN32)
  target_link_libraries(amdhip64 PRIVATE advapi32)
else()
  if(NOT TARGET tracelogging::tracelogging)
    # The installed tracelogging package links its helper library to
    # lttng-ust-tracepoint::lttng-ust-tracepoint but does not install the module that defines
    # that target, so a consumer has to.
    if(NOT TARGET lttng-ust-tracepoint::lttng-ust-tracepoint)
      find_library(CLR_LTTNG_UST_TRACEPOINT_LIBRARY NAMES lttng-ust-tracepoint REQUIRED)
      add_library(lttng-ust-tracepoint::lttng-ust-tracepoint UNKNOWN IMPORTED)
      set_target_properties(lttng-ust-tracepoint::lttng-ust-tracepoint PROPERTIES
                            IMPORTED_LOCATION "${CLR_LTTNG_UST_TRACEPOINT_LIBRARY}")
    endif()
    find_package(tracelogging CONFIG REQUIRED)
  endif()
  target_link_libraries(amdhip64 PRIVATE tracelogging::tracelogging)
endif()

target_sources(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_trace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_trace_probes.cpp)

# The probe schema, generated with the probes: every rocm_hip event field with its C type, enum
# names and capture rule, for tools that read the trace (shared/rocm-trace/scripts/README.md).
# Installed with the runtime it describes.
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/hip_trace_probes.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/rocm-trace
  RENAME rocm_hip.json
  COMPONENT binary)
target_include_directories(amdhip64 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(amdhip64 PRIVATE rocm::trace)
target_compile_definitions(amdhip64 PRIVATE HIP_ROCM_TRACE=1)

# Regenerate the probes in the binary tree and fail if the checked-in files differ. Needs Python
# with shared/rocm-trace/scripts/requirements.txt and a clang resource directory.
option(CLR_ROCM_TRACE_CODEGEN_CHECK "Check that the checked-in rocm_hip probes are current" OFF)
if(CLR_ROCM_TRACE_CODEGEN_CHECK)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(CLR_ROCM_TRACE_RESOURCE_DIR "" CACHE PATH
      "clang resource directory for the probe generator (clang -print-resource-dir)")
  if(NOT CLR_ROCM_TRACE_RESOURCE_DIR)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
                    OUTPUT_VARIABLE _rocm_trace_resource_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(CLR_ROCM_TRACE_RESOURCE_DIR "${_rocm_trace_resource_dir}")
  endif()
  add_custom_target(
    rocm_trace_hip_codegen_check
    COMMAND ${Python3_EXECUTABLE} ${CLR_ROCM_TRACE_DIR}/scripts/rocm_trace_gen.py
            --config ${CMAKE_CURRENT_SOURCE_DIR}/rocm_trace_hip.yaml
            --output-dir ${CMAKE_CURRENT_SOURCE_DIR} --check
            -I ${PROJECT_BINARY_DIR}/include -I ${CMAKE_CURRENT_SOURCE_DIR}/../../include
            -I ${HIP_COMMON_INCLUDE_DIR} -D __HIP_PLATFORM_AMD__
            --resource-dir ${CLR_ROCM_TRACE_RESOURCE_DIR}
    COMMENT "Checking the generated rocm_hip probes"
    VERBATIM)
  add_dependencies(amdhip64 rocm_trace_hip_codegen_check)
endif()
