# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

set(OPENMP_DIRECTIVE_USE_OFFLOAD TRUE)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    if("fortran" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        return()
    endif()
    if("offload" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES)
        set(OPENMP_DIRECTIVE_USE_OFFLOAD FALSE)
    endif()
endif()

rocprofiler_systems_message(STATUS "Configure OpenMP Fortran examples...")

if(NOT ROCPROFSYS_GFX_TARGETS AND OPENMP_DIRECTIVE_USE_OFFLOAD)
    rocprofiler_systems_message(WARNING "No GPU targets detected/set. Disabling OpenMP offload Fortran examples...")
    set(OPENMP_DIRECTIVE_USE_OFFLOAD FALSE)
endif()

# ---------------------------------------------------------------------------------------#
# Compiler setup
# ---------------------------------------------------------------------------------------#

find_program(
    amdflang_EXECUTABLE
    NAMES amdflang
    HINTS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin llvm/bin
)
mark_as_advanced(amdflang_EXECUTABLE)

if(NOT amdflang_EXECUTABLE)
    rocprofiler_systems_message(
      FATAL_ERROR
      "Could not find amdflang. This is required for the OpenMP Fortran examples."
    )
endif()

execute_process(
    COMMAND ${amdflang_EXECUTABLE} --version
    OUTPUT_VARIABLE AMDFLANG_VERSION_CMD_OUTPUT
    RESULT_VARIABLE AMDFLANG_VERSION_CMD_RESULT
)

set(AMDFLANG_FULL_VERSION "")

if(AMDFLANG_VERSION_CMD_RESULT EQUAL 0)
    string(
        REGEX MATCH
        "version ([0-9]+)\\.([0-9]+)\\.([0-9]+)"
        VERSION_MATCH
        "${AMDFLANG_VERSION_CMD_OUTPUT}"
    )
    if(VERSION_MATCH)
        set(AMDFLANG_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(AMDFLANG_VERSION_MINOR "${CMAKE_MATCH_2}")
        set(AMDFLANG_VERSION_PATCH "${CMAKE_MATCH_3}")
        set(AMDFLANG_FULL_VERSION
            "${AMDFLANG_VERSION_MAJOR}.${AMDFLANG_VERSION_MINOR}.${AMDFLANG_VERSION_PATCH}"
        )
        rocprofiler_systems_message(STATUS "Detected amdflang version: ${AMDFLANG_FULL_VERSION}")
    endif()
endif()

if(NOT AMDFLANG_FULL_VERSION)
    rocprofiler_systems_message(FATAL_ERROR "Failed to detect amdflang version.")
endif()

if(AMDFLANG_VERSION_MAJOR LESS 20)
    rocprofiler_systems_message(
        WARNING
        "amdflang version ${AMDFLANG_FULL_VERSION} is less than the minimum required version 20 for OpenMP fortran examples"
    )
    return()
endif()

set(CMAKE_Fortran_COMPILER ${amdflang_EXECUTABLE})
enable_language(Fortran)

# ---------------------------------------------------------------------------------------#
# Compile examples
# ---------------------------------------------------------------------------------------#

set(HOST_EXAMPLE_NAME "openmp-fortran-host")
set(HOST_EXAMPLE_FLAGS "-fopenmp")

add_executable(${HOST_EXAMPLE_NAME})
# TODO: Once SDK issue is fixed (early_fulfill arriving after task_complete causes failure),
# uncomment the call to run_task_detach_phase in host.f90
target_sources(${HOST_EXAMPLE_NAME} PRIVATE host.f90)
target_compile_options(${HOST_EXAMPLE_NAME} PRIVATE ${HOST_EXAMPLE_FLAGS})
target_link_options(${HOST_EXAMPLE_NAME} PRIVATE ${HOST_EXAMPLE_FLAGS})

if(OPENMP_DIRECTIVE_USE_OFFLOAD)
    set(OFFLOAD_EXAMPLE_NAME "openmp-fortran-offload")

    add_executable(${OFFLOAD_EXAMPLE_NAME})
    target_sources(${OFFLOAD_EXAMPLE_NAME} PRIVATE offload.f90)
    target_compile_options(${OFFLOAD_EXAMPLE_NAME} PRIVATE ${HOST_EXAMPLE_FLAGS})
    target_link_options(${OFFLOAD_EXAMPLE_NAME} PRIVATE ${HOST_EXAMPLE_FLAGS})

    foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
        target_compile_options(${OFFLOAD_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
        target_link_options(${OFFLOAD_EXAMPLE_NAME} PRIVATE --offload-arch=${arch})
    endforeach()
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES)
    foreach(_TARGET ${HOST_EXAMPLE_NAME} ${OFFLOAD_EXAMPLE_NAME})
        if(TARGET ${_TARGET})
            install(
                TARGETS ${_TARGET}
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
                COMPONENT rocprofiler-systems-examples
            )
        endif()
    endforeach()
endif()
