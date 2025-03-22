#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-tests-bin-transpose LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

option(TRANSPOSE_USE_MPI "Enable MPI support in transpose exe" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(transpose.cpp PROPERTIES LANGUAGE HIP)

function(transpose_build_target _NAME _COMPILE_DEFS _LINK_TARGETS)
    add_executable(${_NAME})
    target_sources(${_NAME} PRIVATE transpose.cpp)
    target_compile_definitions(${_NAME} PRIVATE ${_COMPILE_DEFS})
    target_compile_options(${_NAME} PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow -Werror)
    target_link_libraries(
        ${_NAME} PRIVATE Threads::Threads rocprofiler-sdk::tests-build-flags
                         ${_LINK_TARGETS})

    if(TRANSPOSE_USE_MPI)
        find_package(MPI REQUIRED)
        target_compile_definitions(${_NAME} PRIVATE USE_MPI=1)
        target_link_libraries(${_NAME} PRIVATE MPI::MPI_C)
    endif()

endfunction()

find_package(Threads REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

transpose_build_target(transpose "" rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)

#
# using old roctracer roctx
#
find_path(
    roctracer_roctx_ROOT_DIR
    NAMES include/roctracer/roctx.h
          lib/${CMAKE_SHARED_LIBRARY_PREFIX}roctx64${CMAKE_SHARED_LIBRARY_SUFFIX}
    HINTS ${hip_DIR} ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${hip_DIR} ${ROCM_PATH} ENV ROCM_PATH /opt/rocm)

find_path(
    roctracer_roctx_INCLUDE_DIR
    NAMES roctracer/roctx.h
    HINTS ${roctracer_roctx_ROOT_DIR}
    PATHS ${roctracer_roctx_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(
    roctracer_roctx_LIBRARY
    NAMES roctx64
    HINTS ${roctracer_roctx_ROOT_DIR}
    PATHS ${roctracer_roctx_ROOT_DIR}
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(roctracer-roctx DEFAULT_MSG roctracer_roctx_ROOT_DIR
                                  roctracer_roctx_INCLUDE_DIR roctracer_roctx_LIBRARY)

if(roctracer-roctx_FOUND)
    add_library(roctracer-roctx INTERFACE IMPORTED)
    add_library(roctracer-roctx::roctracer-roctx ALIAS roctracer-roctx)
    target_include_directories(roctracer-roctx INTERFACE ${roctracer_roctx_INCLUDE_DIR})
    target_link_libraries(roctracer-roctx INTERFACE ${roctracer_roctx_LIBRARY})
endif()

if(TARGET roctracer-roctx::roctracer-roctx)
    transpose_build_target(transpose-roctracer-roctx "USE_ROCTRACER_ROCTX=1"
                           roctracer-roctx::roctracer-roctx)
endif()
