#
#
#
cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

# try to find a compatible HIP version
if(ROCmVersion_MAJOR_VERSION AND ROCmVersion_MAJOR_VERSION GREATER_EQUAL 6)
    find_package(hip ${ROCmVersion_MAJOR_VERSION}.0.0)
else()
    find_package(hip)
endif()

if(NOT hip_FOUND)
    message(WARNING "ROCm >= 5.6 not found. Skipping OpenMP target example.")
    return()
elseif(hip_FOUND AND hip_VERSION VERSION_LESS 5.6.0)
    message(
        WARNING
            "ROCm >= 5.6 not found (found ${hip_VERSION}). Skipping OpenMP target example."
        )
    return()
endif()

if(NOT OMP_TARGET_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(OMP_TARGET_COMPILER
            "${amdclangpp_EXECUTABLE}"
            CACHE FILEPATH "OpenMP target compiler")
    else()
        message(WARNING "OpenMP target compiler not found. Skipping this example.")
        return()
    endif()
endif()

project(rocprofiler-systems-example-openmp-target-lib LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

set(DEFAULT_GPU_TARGETS
    "gfx900"
    "gfx906"
    "gfx908"
    "gfx90a"
    "gfx940"
    "gfx941"
    "gfx942"
    "gfx1030"
    "gfx1010"
    "gfx1100"
    "gfx1101"
    "gfx1102")

set(GPU_TARGETS
    "${DEFAULT_GPU_TARGETS}"
    CACHE STRING "GPU targets to compile for")

find_package(Threads REQUIRED)

add_library(openmp-target-lib SHARED)
target_sources(openmp-target-lib PRIVATE library.cpp)
target_link_libraries(openmp-target-lib PUBLIC Threads::Threads)
target_compile_options(openmp-target-lib PRIVATE -fopenmp -ggdb)
target_link_options(openmp-target-lib PUBLIC -fopenmp)

foreach(_TARGET ${GPU_TARGETS})
    target_compile_options(openmp-target-lib PRIVATE --offload-arch=${_TARGET})
    target_link_options(openmp-target-lib PUBLIC --offload-arch=${_TARGET})
endforeach()

message(STATUS "Using OpenMP target compiler: ${OMP_TARGET_COMPILER}")

get_filename_component(OMP_TARGET_COMPILER_DIR ${OMP_TARGET_COMPILER} PATH)
get_filename_component(OMP_TARGET_COMPILER_DIR ${OMP_TARGET_COMPILER_DIR} PATH)

message(STATUS "Using OpemMP target compiler directory: ${OMP_TARGET_COMPILER_DIR}")

if(NOT EXISTS ${OMP_TARGET_COMPILER_DIR}/llvm/lib)
    message(FATAL_ERROR "${OMP_TARGET_COMPILER_DIR}/llvm/lib does not exist")
endif()
set_target_properties(
    openmp-target-lib
    PROPERTIES BUILD_RPATH
               "${OMP_TARGET_COMPILER_DIR}/llvm/lib:${OMP_TARGET_COMPILER_DIR}/lib"
               OUTPUT_NAME "openmp-target"
               POSITION_INDEPENDENT_CODE ON)

rocprofiler_systems_custom_compilation(TARGET openmp-target-lib
                                       COMPILER ${OMP_TARGET_COMPILER})

add_executable(openmp-target)
target_sources(openmp-target PRIVATE main.cpp)
target_link_libraries(openmp-target PRIVATE openmp-target-lib)
target_compile_options(openmp-target PRIVATE -ggdb)

set_target_properties(
    openmp-target
    PROPERTIES BUILD_RPATH
               "${OMP_TARGET_COMPILER_DIR}/llvm/lib:${OMP_TARGET_COMPILER_DIR}/lib"
               POSITION_INDEPENDENT_CODE ON)

rocprofiler_systems_custom_compilation(TARGET openmp-target
                                       COMPILER ${OMP_TARGET_COMPILER})
