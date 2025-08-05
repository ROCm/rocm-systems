cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

if(NOT OMP_TARGET_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH}
        ENV ROCM_PATH
        /opt/rocm
        PATHS ${ROCM_PATH}
        ENV ROCM_PATH
        /opt/rocm
        PATH_SUFFIXES bin llvm/bin
    )
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(OMP_TARGET_COMPILER
            "${amdclangpp_EXECUTABLE}"
            CACHE FILEPATH
            "OpenMP target compiler"
        )
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
    "gfx950"
    "gfx1030"
    "gfx1010"
    "gfx1100"
    "gfx1101"
    "gfx1102"
)

set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "GPU targets to compile for")

find_package(Threads REQUIRED)

function(add_offload_flags tgt)
    foreach(arch IN LISTS GPU_TARGETS)
        target_compile_options(${tgt} PRIVATE --offload-arch=${arch})
        target_link_options(${tgt} PUBLIC --offload-arch=${arch})
    endforeach()
endfunction()

get_filename_component(OMP_TARGET_COMPILER_DIR ${OMP_TARGET_COMPILER} PATH)
get_filename_component(OMP_TARGET_COMPILER_DIR ${OMP_TARGET_COMPILER_DIR} PATH)

message(STATUS "Using OpenMP target compiler: ${OMP_TARGET_COMPILER}")
message(STATUS "Using OpenMP target compiler directory: ${OMP_TARGET_COMPILER_DIR}")

set(_rocm_llvm_lib "${OMP_TARGET_COMPILER_DIR}/llvm/lib")
set(_rocm_clang_lib "${OMP_TARGET_COMPILER_DIR}/lib")
set(_COMMON_RPATH "${_rocm_llvm_lib};${_rocm_clang_lib}")

if(NOT EXISTS "${_rocm_llvm_lib}/libomptarget.so")
    message(FATAL_ERROR "Could not find libomptarget.so in ${_rocm_llvm_lib}")
endif()

# Shared library

add_library(openmp-target-lib SHARED)
target_sources(openmp-target-lib PRIVATE library.cpp)
target_link_libraries(openmp-target-lib PUBLIC Threads::Threads)
target_compile_options(openmp-target-lib PRIVATE -fopenmp -ggdb)
target_link_options(openmp-target-lib PUBLIC -fopenmp)
add_offload_flags(openmp-target-lib)

# Executable

add_executable(openmp-target)
target_sources(openmp-target PRIVATE main.cpp)
target_link_libraries(openmp-target PRIVATE openmp-target-lib)
target_compile_options(openmp-target PRIVATE -fopenmp -ggdb)
target_link_options(openmp-target PUBLIC -fopenmp)
add_offload_flags(openmp-target)

foreach(tgt openmp-target-lib openmp-target)
    set_target_properties(
        ${tgt}
        PROPERTIES
            BUILD_RPATH "${_COMMON_RPATH}"
            INSTALL_RPATH "${_COMMON_RPATH}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            POSITION_INDEPENDENT_CODE ON
    )
    target_link_options(
        ${tgt}
        PUBLIC
            "-L${_rocm_llvm_lib}"
            "-Wl,-rpath,${_rocm_llvm_lib}"
            "-Wl,-rpath,${_rocm_clang_lib}"
    )
endforeach()

rocprofiler_systems_custom_compilation(
    TARGET   openmp-target-lib
    COMPILER ${OMP_TARGET_COMPILER}
)

rocprofiler_systems_custom_compilation(TARGET openmp-target
                                       COMPILER ${OMP_TARGET_COMPILER}
)
